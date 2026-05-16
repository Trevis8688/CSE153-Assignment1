# CSE 153/253 Assignment 1 - Trevor Duong
# Training code for all three tasks. Running this file regenerates
# predictions1.json, predictions2.json and predictions3.json.

import os
import random

import urllib.request
from collections import Counter, defaultdict

import numpy as np
import miditoolkit
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
from transformers import BertConfig, BertModel


# Task 1: Composer classification via REMI Transformer

T1_DATAROOT = "student_files/task1_composer_classification"
T1_TRANSPOSITIONS = list(range(-5, 7))   # token-space pitch shift for training augmentation
T1_MAX_LEN = 512
T1_STRIDE = 256                          # window stride at inference time
T1_BATCH_SIZE = 32
T1_EPOCHS = 30
T1_LR = 1e-4
T1_WEIGHT_DECAY = 0.01
T1_N_CLASSES = 8
T1_WINDOWS_PER_TRAIN_MIDI = 4            # random training windows seen per MIDI per epoch
T1_DROPOUT = 0.1


def t1_build_tokenizer():
    cfg = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False, use_rests=False,
        use_tempos=True, num_tempos=32, tempo_range=(40, 250),
        use_time_signatures=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
    )
    return REMI(cfg)


def t1_midi_to_ids(tokenizer, path):
    # miditok 3.x returns a list of TokSequence (one per track); concatenate so each MIDI is one stream
    seq = tokenizer(os.path.join(T1_DATAROOT, path))
    if isinstance(seq, list):
        ids = []
        for s in seq:
            ids.extend(s.ids)
    else:
        ids = list(seq.ids)
    return np.array(ids, dtype=np.int64)


def t1_make_transpose_map(tokenizer, semitones, vocab_size):
    # transposition is implemented as a per-token-id remap: every Pitch_X token id is rewritten
    # to the Pitch_(X+semitones) id, leaving Bar / Position / Velocity / Duration / etc unchanged
    mp = np.arange(vocab_size, dtype=np.int64)
    for tok_str, tok_id in tokenizer.vocab.items():
        if not tok_str.startswith("Pitch_"):
            continue
        try:
            pitch = int(tok_str.split("_")[1])
        except ValueError:
            continue
        target = f"Pitch_{pitch + semitones}"
        if target in tokenizer.vocab:
            mp[tok_id] = tokenizer.vocab[target]
    return mp


def t1_pack_window(window_ids, pad_id, bos_id):
    # prepend BOS as a sentinel and right-pad to T1_MAX_LEN; attention mask zeroes out pads
    inner = window_ids[: T1_MAX_LEN - 1]
    ids = np.full(T1_MAX_LEN, pad_id, dtype=np.int64)
    ids[0] = bos_id
    ids[1:1 + len(inner)] = inner
    attn = np.zeros(T1_MAX_LEN, dtype=np.int64)
    attn[: 1 + len(inner)] = 1
    return ids, attn


def t1_enumerate_windows(tok_ids):
    inner = T1_MAX_LEN - 1
    L = len(tok_ids)
    if L <= inner:
        return [tok_ids]
    starts = list(range(0, L - inner + 1, T1_STRIDE))
    if starts[-1] + inner < L:
        starts.append(L - inner)
    return [tok_ids[s:s + inner] for s in starts]


class T1TrainSet(Dataset):
    # each draw returns a random window from a random MIDI with a random transposition
    def __init__(self, tokens_list, labels, transpose_maps, pad_id, bos_id,
                 n_per_midi=T1_WINDOWS_PER_TRAIN_MIDI):
        self.tokens = tokens_list
        self.labels = labels
        self.transpose_maps = transpose_maps
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.n = n_per_midi

    def __len__(self):
        return len(self.tokens) * self.n

    def __getitem__(self, idx):
        midi_idx = idx % len(self.tokens)
        tok = self.tokens[midi_idx]
        semi = random.choice(T1_TRANSPOSITIONS)
        tok = self.transpose_maps[semi][tok]
        inner = T1_MAX_LEN - 1
        if len(tok) <= inner:
            window = tok
        else:
            start = random.randint(0, len(tok) - inner)
            window = tok[start:start + inner]
        ids, attn = t1_pack_window(window, self.pad_id, self.bos_id)
        return torch.from_numpy(ids), torch.from_numpy(attn), int(self.labels[midi_idx])


class T1EvalSet(Dataset):
    # enumerate every window for every MIDI; aggregation by path is done at scoring time
    def __init__(self, tokens_list, paths, pad_id, bos_id):
        self.items = []
        self.pad_id = pad_id
        self.bos_id = bos_id
        for tok, p in zip(tokens_list, paths):
            for w in t1_enumerate_windows(tok):
                self.items.append((w, p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        window, p = self.items[idx]
        ids, attn = t1_pack_window(window, self.pad_id, self.bos_id)
        return torch.from_numpy(ids), torch.from_numpy(attn), p


class RemiBert(nn.Module):
    # small BERT encoder over REMI tokens, mean-pooled over non-pad positions into a class head
    def __init__(self, vocab_size, n_classes, pad_id, max_len=T1_MAX_LEN,
                 d_model=256, n_layers=6, n_heads=8, ff=1024, dropout=T1_DROPOUT):
        super().__init__()
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=ff,
            max_position_embeddings=max_len,
            type_vocab_size=1,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            pad_token_id=pad_id,
        )
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.dropout(pooled))


def t1_aggregate_predictions(model, loader, device):
    model.eval()
    logits_by_path = defaultdict(list)
    with torch.no_grad():
        for ids, attn, paths in loader:
            ids = ids.to(device); attn = attn.to(device)
            logits = model(ids, attn).cpu().numpy()
            for i, p in enumerate(paths):
                logits_by_path[p].append(logits[i])
    return {p: np.mean(np.stack(ls), axis=0) for p, ls in logits_by_path.items()}


def t1_get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_task1():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = t1_get_device()
    print("[task1] device:", device)

    tokenizer = t1_build_tokenizer()
    vocab_size = len(tokenizer)
    pad_id = tokenizer["PAD_None"]
    bos_id = tokenizer["BOS_None"]

    train_meta = eval(open(os.path.join(T1_DATAROOT, "train.json")).read())
    test_list = eval(open(os.path.join(T1_DATAROOT, "test.json")).read())
    train_paths = list(train_meta.keys())
    train_labels = np.array([int(train_meta[p]) for p in train_paths])
    test_paths = list(test_list)

    tokens = {}
    for p in tqdm(train_paths + test_paths, desc="[task1] tokenizing"):
        tokens[p] = t1_midi_to_ids(tokenizer, p)

    transpose_maps = {s: t1_make_transpose_map(tokenizer, s, vocab_size) for s in T1_TRANSPOSITIONS}

    # 90/10 stratified split on original pieces; no augmentation in the val fold
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_paths)), test_size=0.1, random_state=0, stratify=train_labels,
    )
    tr_tokens = [tokens[train_paths[i]] for i in tr_idx]
    tr_labels = train_labels[tr_idx]
    va_tokens = [tokens[train_paths[i]] for i in va_idx]
    va_paths = [train_paths[i] for i in va_idx]
    va_labels = train_labels[va_idx]
    te_tokens = [tokens[p] for p in test_paths]

    train_set = T1TrainSet(tr_tokens, tr_labels, transpose_maps, pad_id, bos_id)
    val_set = T1EvalSet(va_tokens, va_paths, pad_id, bos_id)
    test_set = T1EvalSet(te_tokens, test_paths, pad_id, bos_id)
    loader_tr = DataLoader(train_set, batch_size=T1_BATCH_SIZE, shuffle=True)
    loader_va = DataLoader(val_set, batch_size=T1_BATCH_SIZE, shuffle=False)
    loader_te = DataLoader(test_set, batch_size=T1_BATCH_SIZE, shuffle=False)
    val_label_lookup = {p: l for p, l in zip(va_paths, va_labels)}

    model = RemiBert(vocab_size, T1_N_CLASSES, pad_id).to(device)
    print(f"[task1] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # inverse-frequency class weights handle the 13x imbalance (class 1 has ~490, class 7 has ~37)
    cls_counts = Counter(int(c) for c in tr_labels)
    weights = np.array([1.0 / cls_counts[i] for i in range(T1_N_CLASSES)], dtype=np.float32)
    weights = weights / weights.mean()
    cls_weight = torch.tensor(weights, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=T1_LR, weight_decay=T1_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=cls_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T1_EPOCHS)

    best_acc = -1.0
    best_state = None
    for epoch in range(T1_EPOCHS):
        model.train()
        running, nb = 0.0, 0
        for ids, attn, y in tqdm(loader_tr, desc=f"[task1] epoch {epoch+1}/{T1_EPOCHS}"):
            ids = ids.to(device); attn = attn.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, attn), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item(); nb += 1
        scheduler.step()

        val_logits = t1_aggregate_predictions(model, loader_va, device)
        correct = sum(int(np.argmax(val_logits[p])) == int(val_label_lookup[p]) for p in va_paths)
        val_acc = correct / max(len(va_paths), 1)
        print(f"[task1 ep{epoch+1}] loss={running/nb:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[task1] best val acc: {best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_logits = t1_aggregate_predictions(model, loader_te, device)
    pred_dict = {p: int(np.argmax(test_logits[p])) for p in test_paths}
    with open("predictions1.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task1] wrote predictions1.json ({len(pred_dict)} entries)")


# ============================ Task 2: Temporal order prediction ============================

T2_DATAROOT = "student_files/task2_next_sequence_prediction"


def segment_summary(path, cache):
    if path in cache:
        return cache[path]
    midi = miditoolkit.MidiFile(os.path.join(T2_DATAROOT, path))
    tpq = midi.ticks_per_beat or 480
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    if not notes:
        cache[path] = {"empty": True}
        return cache[path]

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    starts = np.array([n.start for n in notes], dtype=np.float64) / tpq
    ends = np.array([n.end for n in notes], dtype=np.float64) / tpq
    vels = np.array([n.velocity for n in notes], dtype=np.float64)

    pc = np.zeros(12)
    for p in pitches:
        pc[int(p) % 12] += 1
    pc /= pc.sum() + 1e-9

    # describe the first/last few notes so we can reason about the seam between segments
    K = 5
    order = np.argsort(starts)
    head_idx = order[:K]
    tail_idx = order[-K:]
    first_idx = int(np.argmin(starts))
    last_idx = int(np.argmax(starts))

    cache[path] = {
        "empty": False,
        "num_notes": len(notes),
        "pitch_mean": float(pitches.mean()),
        "pitch_std": float(pitches.std()),
        "pitch_min": float(pitches.min()),
        "pitch_max": float(pitches.max()),
        "vel_mean": float(vels.mean()),
        "vel_std": float(vels.std()),
        "pc_hist": pc,
        "head_pitch_mean": float(pitches[head_idx].mean()),
        "head_pitch_std": float(pitches[head_idx].std()),
        "tail_pitch_mean": float(pitches[tail_idx].mean()),
        "tail_pitch_std": float(pitches[tail_idx].std()),
        "head_vel_mean": float(vels[head_idx].mean()),
        "tail_vel_mean": float(vels[tail_idx].mean()),
        "first_pitch": float(pitches[first_idx]),
        "last_pitch": float(pitches[last_idx]),
        "first_vel": float(vels[first_idx]),
        "last_vel": float(vels[last_idx]),
        "total_beats": float(ends.max() - starts.min()),
        "duration_first_note": float(ends[first_idx] - starts[first_idx]),
        "duration_last_note": float(ends[last_idx] - starts[last_idx]),
    }
    return cache[path]


def pair_features(s1, s2):
    # features describing the ordering (s1, s2): compare the s1->s2 seam against s2->s1
    if s1.get("empty") or s2.get("empty"):
        return np.zeros(40, dtype=np.float64)

    fwd_pitch = s2["first_pitch"] - s1["last_pitch"]
    rev_pitch = s1["first_pitch"] - s2["last_pitch"]
    fwd_tail_head = s2["head_pitch_mean"] - s1["tail_pitch_mean"]
    rev_tail_head = s1["head_pitch_mean"] - s2["tail_pitch_mean"]
    fwd_vel = s2["first_vel"] - s1["last_vel"]
    rev_vel = s1["first_vel"] - s2["last_vel"]

    a, b = s1["pc_hist"], s2["pc_hist"]
    pc_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    return np.array([
        fwd_pitch, abs(fwd_pitch), fwd_tail_head, fwd_vel, pc_sim,
        rev_pitch, abs(rev_pitch), rev_tail_head, rev_vel,
        abs(fwd_pitch) - abs(rev_pitch), fwd_tail_head - rev_tail_head, fwd_vel - rev_vel,
        s1["pitch_mean"] - s2["pitch_mean"], s1["pitch_std"] - s2["pitch_std"],
        s1["vel_mean"] - s2["vel_mean"], s1["num_notes"] - s2["num_notes"],
        s1["total_beats"] - s2["total_beats"],
        s1["last_pitch"], s2["first_pitch"], s1["first_pitch"], s2["last_pitch"],
        s1["tail_pitch_mean"], s2["head_pitch_mean"],
        s1["head_pitch_mean"], s2["tail_pitch_mean"],
        s1["pitch_mean"], s2["pitch_mean"], s1["pitch_std"], s2["pitch_std"],
        s1["vel_mean"], s2["vel_mean"], s1["num_notes"], s2["num_notes"],
        s1["total_beats"], s2["total_beats"],
        s1["duration_last_note"], s2["duration_first_note"],
        s1["last_vel"], s2["first_vel"], pc_sim,
    ], dtype=np.float64)


def run_task2():
    train = eval(open(os.path.join(T2_DATAROOT, "train.json")).read())
    test = eval(open(os.path.join(T2_DATAROOT, "test.json")).read())
    train_pairs = list(train.keys())
    train_labels = np.array([1 if train[k] else 0 for k in train_pairs])
    test_pairs = list(test)

    cache = {}

    def build(pairs, desc):
        rows = []
        for p1, p2 in tqdm(pairs, desc=desc):
            rows.append(pair_features(segment_summary(p1, cache), segment_summary(p2, cache)))
        return np.array(rows)

    X_train = build(train_pairs, "task2 train")
    X_test = build(test_pairs, "task2 test")

    # the task is symmetric: swapping a pair flips its label, so add the swaps as extra data
    X_swap = build([(p2, p1) for p1, p2 in train_pairs], "task2 swapped")
    X_all = np.nan_to_num(np.concatenate([X_train, X_swap]), 0.0, 0.0, 0.0)
    y_all = np.concatenate([train_labels, 1 - train_labels])
    X_test = np.nan_to_num(X_test, 0.0, 0.0, 0.0)

    models = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
        "gb": GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=0),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, m in models.items():
        scores = cross_val_score(m, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"[task2] {name}: CV acc = {scores.mean():.4f} +/- {scores.std():.4f}")

    fitted = {name: m.fit(X_all, y_all) for name, m in models.items()}
    probs = np.mean([m.predict_proba(X_test) for m in fitted.values()], axis=0)
    classes = fitted["gb"].classes_
    preds = classes[np.argmax(probs, axis=1)]

    pred_dict = {tuple(test_pairs[i]): bool(preds[i] == 1) for i in range(len(test_pairs))}
    with open("predictions2.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task2] wrote predictions2.json ({len(pred_dict)} entries)")


# Task 3: Audio tagging via fine-tuned PANNs CNN14

T3_DATAROOT = "student_files/task3_audio_classification"
SAMPLE_RATE = 32000           # PANNs native sample rate; the backbone expects 32 kHz raw audio
AUDIO_DURATION = 10
N_CLASSES = 10
T3_BATCH_SIZE = 32
T3_EPOCHS = 25
BACKBONE_LR = 1e-4            # small nudge on the pretrained backbone
HEAD_LR = 1e-3                # larger step on the fresh classifier head
MIXUP_ALPHA = 0.4
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance', 'blues', 'punk', 'chill', 'electronic', 'country']
CNN14_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CNN14_LOCAL = "Cnn14_mAP=0.431.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_waveform(path):
    # PANNs takes mono audio at 32 kHz, padded or truncated to a fixed 10-second window.
    # The model has its own log-mel front-end, so no spectrogram computation is needed here.
    waveform, _ = librosa.load(os.path.join(T3_DATAROOT, path), sr=SAMPLE_RATE)
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return torch.from_numpy(waveform.astype(np.float32))


class TaggingDataset(Dataset):
    # preloads all waveforms in memory, the dataset is small enough that this is faster than
    # re-decoding each clip every epoch
    def __init__(self, meta):
        self.meta = meta
        self.ids = list(meta.keys())
        self.cache = {}
        for path in tqdm(self.ids, desc="Preloading waveforms"):
            self.cache[path] = load_waveform(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = self.ids[idx]
        label = torch.tensor([1 if t in self.meta[path] else 0 for t in TAGS], dtype=torch.float32)
        return self.cache[path], label, path


def download_cnn14_checkpoint():
    # The published PANNs CNN14 checkpoint from Zenodo, around 326 MB.
    if not os.path.exists(CNN14_LOCAL):
        print("[task3] downloading CNN14 checkpoint ...")
        urllib.request.urlretrieve(CNN14_URL, CNN14_LOCAL)
    return CNN14_LOCAL


class Cnn14Tagger(nn.Module):
    # PANNs CNN14 backbone with a fresh classifier head on top of its 2048-d embedding.
    def __init__(self, ckpt_path, n_classes=N_CLASSES):
        super().__init__()
        from panns_inference.models import Cnn14
        self.backbone = Cnn14(sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320,
                              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state["model"])
        # the original 527-class fc_audioset is ignored; we take the embedding before it instead
        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        # x: (batch, samples) raw waveform at 32 kHz
        out = self.backbone(x)
        return self.head(out["embedding"])


def mixup_batch(x, y, alpha):
    # blend two random samples in each batch and blend their multi-hot labels the same way
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_targets.append(y)
            all_paths += list(paths)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.sigmoid(logits).numpy()
    mAP = None
    if targets.sum() > 0:
        try:
            mAP = average_precision_score(targets, probs, average="macro")
        except Exception:
            mAP = None
    return probs, all_paths, mAP


def run_task3():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = get_device()
    print("[task3] device:", device)

    train_meta = eval(open(os.path.join(T3_DATAROOT, "train.json")).read())
    test_meta = {k: [] for k in eval(open(os.path.join(T3_DATAROOT, "test.json")).read())}

    full_train = TaggingDataset(train_meta)
    g = torch.Generator().manual_seed(0)
    n_train = int(len(full_train) * 0.9)
    n_valid = len(full_train) - n_train
    train_sub, valid_sub = random_split(full_train, [n_train, n_valid], generator=g)
    test_set = TaggingDataset(test_meta)

    loader_train = DataLoader(train_sub, batch_size=T3_BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(valid_sub, batch_size=T3_BATCH_SIZE, shuffle=False)
    loader_test = DataLoader(test_set, batch_size=T3_BATCH_SIZE, shuffle=False)

    ckpt = download_cnn14_checkpoint()
    model = Cnn14Tagger(ckpt).to(device)
    # different learning rates for the pretrained backbone and the fresh head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(),     "lr": HEAD_LR},
    ], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T3_EPOCHS)

    best_map = -1.0
    best_state = None
    for epoch in range(T3_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y, _ in tqdm(loader_train, desc=f"Epoch {epoch+1}/{T3_EPOCHS}"):
            x = x.to(device); y = y.to(device)
            x_mix, y_mix = mixup_batch(x, y, MIXUP_ALPHA)
            optimizer.zero_grad()
            loss = criterion(model(x_mix), y_mix)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        _, _, val_map = evaluate(model, loader_valid, device)
        print(f"[task3 ep{epoch+1}] loss={running_loss/n_batches:.4f} val_mAP={val_map:.4f}")
        if val_map is not None and val_map > best_map:
            best_map = val_map
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[task3] best val mAP: {best_map:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    probs, paths, _ = evaluate(model, loader_test, device)
    pred_dict = {}
    for i, p in enumerate(paths):
        key = p[2:] if p.startswith("./") else p
        pred_dict[key] = {TAGS[j]: float(probs[i][j]) for j in range(N_CLASSES)}

    with open("predictions3.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task3] wrote predictions3.json ({len(pred_dict)} entries)")


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
