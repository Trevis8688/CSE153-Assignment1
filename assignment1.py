# CSE 153/253 Assignment 1 - Trevor Duong
# Training code for all three tasks. Running this file regenerates
# predictions1.json, predictions2.json and predictions3.json.

import os
import random

import numpy as np
import miditoolkit
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score
from tqdm import tqdm


# Task 1: Composer classification

T1_DATAROOT = "student_files/task1_composer_classification"
TRANSPOSITIONS = list(range(-5, 7))  # -5..+6 semitones; augments training to make the model key-invariant


def stats(arr):
    # mean / std / min / max / median, with a safe fallback for empty inputs
    if len(arr) == 0:
        return [0.0] * 5
    a = np.asarray(arr, dtype=np.float64)
    return [float(a.mean()), float(a.std()), float(a.min()), float(a.max()), float(np.median(a))]


def task1_load_notes(path):
    # parse a MIDI once; transposition reuses the parsed notes without re-reading the file
    midi = miditoolkit.MidiFile(os.path.join(T1_DATAROOT, path))
    tpq = midi.ticks_per_beat or 480
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        for inst in midi.instruments:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    tempos = np.array([t.tempo for t in midi.tempo_changes] or [120.0], dtype=np.float64)
    if midi.time_signature_changes:
        ts = (midi.time_signature_changes[0].numerator, midi.time_signature_changes[0].denominator)
    else:
        ts = (4, 4)
    return notes, tpq, tempos, ts


# Feature layout (see assignment of feats below):
#   7 stat blocks * 5 values (pitches, durs, iois, intervals, vels, poly, abs_intervals) = 35
#   pitch-class histogram (12) + key-centered pitch-class histogram (12)
#   register histogram (8)
#   melodic interval histogram (27) + skip-2 interval histogram (27)
#   duration histogram (8)
#   IOI ratio histogram (7)
#   chord_frac (1)
#   tempo stats (4)
#   ts/density/length scalars (5)
T1_FEATURE_DIM = 5 * 7 + 12 + 12 + 8 + 27 + 27 + 8 + 7 + 1 + 4 + 5  # = 146


def task1_features(notes, tpq, tempos, ts, transpose=0):
    if not notes:
        return np.zeros(T1_FEATURE_DIM)

    pitches = np.array([n.pitch + transpose for n in notes], dtype=np.float64)
    durs = np.array([n.end - n.start for n in notes], dtype=np.float64) / tpq
    starts = np.array([n.start for n in notes], dtype=np.float64)
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    iois = np.diff(np.sort(np.unique(starts))) / tpq  # inter-onset intervals
    order = np.argsort(starts)
    ordered_pitch = pitches[order]
    intervals = np.diff(ordered_pitch)  # melodic intervals
    skip2 = ordered_pitch[2:] - ordered_pitch[:-2] if len(ordered_pitch) > 2 else np.array([])

    # pitch-class histogram, plus a transposition-invariant copy (rotated so most common pc is bin 0)
    pc = np.zeros(12)
    for p in pitches:
        pc[int(p) % 12] += 1
    pc /= pc.sum() + 1e-9
    pc_centered = np.roll(pc, -int(np.argmax(pc)))

    reg = np.zeros(8)
    for p in pitches:
        reg[min(max(int(p) // 16, 0), 7)] += 1
    reg /= reg.sum() + 1e-9

    # polyphony profile from a note on/off sweep
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    poly = 0
    poly_profile = []
    for _, delta in events:
        poly += delta
        poly_profile.append(poly)
    poly_profile = np.array(poly_profile or [0.0], dtype=np.float64)

    total_beats = max((n.end for n in notes), default=1) / tpq
    density = len(notes) / max(total_beats, 1e-3)

    feats = []
    feats += stats(pitches)
    feats += stats(durs)
    feats += stats(iois) if len(iois) else [0.0] * 5
    feats += stats(intervals) if len(intervals) else [0.0] * 5
    feats += stats(vels)
    feats += stats(poly_profile)
    feats += list(pc)
    feats += list(pc_centered)
    feats += list(reg)

    # melodic-interval histogram, bucketed to [-13, +13] (transposition-invariant)
    int_hist = np.zeros(27)
    for iv in intervals:
        int_hist[int(np.clip(iv + 13, 0, 26))] += 1
    int_hist /= int_hist.sum() + 1e-9
    feats += list(int_hist)

    # skip-2 interval histogram (transposition-invariant): pitch difference across one intervening note
    skip_hist = np.zeros(27)
    for iv in skip2:
        skip_hist[int(np.clip(iv + 13, 0, 26))] += 1
    skip_hist /= skip_hist.sum() + 1e-9
    feats += list(skip_hist)

    # duration histogram (in beats)
    dur_edges = [0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
    dur_hist = np.zeros(len(dur_edges) - 1)
    for d in durs:
        for i in range(len(dur_edges) - 1):
            if dur_edges[i] <= d < dur_edges[i + 1]:
                dur_hist[i] += 1
                break
    dur_hist /= dur_hist.sum() + 1e-9
    feats += list(dur_hist)

    # absolute interval-size stats (melodic "leapiness")
    abs_int = np.abs(intervals)
    feats += stats(abs_int) if len(abs_int) else [0.0] * 5

    # IOI ratio histogram (tempo-invariant rhythm signature): ratio of consecutive IOIs
    ioi_ratio_hist = np.zeros(7)
    if len(iois) > 1:
        ratios = iois[1:] / (iois[:-1] + 1e-9)
        ratio_edges = [0, 0.33, 0.6, 0.9, 1.1, 1.7, 3.0, 1e9]
        for r in ratios:
            for i in range(len(ratio_edges) - 1):
                if ratio_edges[i] <= r < ratio_edges[i + 1]:
                    ioi_ratio_hist[i] += 1
                    break
        ioi_ratio_hist /= ioi_ratio_hist.sum() + 1e-9
    feats += list(ioi_ratio_hist)

    # fraction of simultaneous note onsets (chordal vs monophonic texture)
    n_onsets = len(np.unique(starts))
    chord_frac = 1.0 - n_onsets / max(len(notes), 1)
    feats += [chord_frac]

    feats += [float(tempos.mean()), float(tempos.std()), float(tempos.min()), float(tempos.max())]
    feats += [float(ts[0]), float(ts[1]), float(density), float(len(notes)), float(total_beats)]
    return np.array(feats, dtype=np.float64)


def _task1_models():
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
        ]),
        "hgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                              max_depth=None, l2_regularization=1.0,
                                              random_state=0),
        "rf": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0,
                                     class_weight="balanced", min_samples_leaf=2),
    }


def run_task1():
    train = eval(open(os.path.join(T1_DATAROOT, "train.json")).read())
    test = eval(open(os.path.join(T1_DATAROOT, "test.json")).read())
    train_paths = list(train.keys())
    train_labels = np.array([int(train[k]) for k in train_paths])
    test_paths = list(test)

    parsed = {}
    for p in tqdm(train_paths + test_paths, desc="task1 parsing midi"):
        parsed[p] = task1_load_notes(p)

    # un-augmented features used for the validation fold (honest CV)
    X_base = np.nan_to_num(np.array([task1_features(*parsed[p]) for p in train_paths]))

    # honest CV: split on original pieces, augment ONLY the training fold so a transposed copy
    # of a piece can never leak into its own validation fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_acc = {name: [] for name in _task1_models()}
    fold_acc["ensemble"] = []
    for tr_idx, va_idx in skf.split(X_base, train_labels):
        Xtr, ytr = [], []
        for i in tr_idx:
            notes, tpq, tempos, ts = parsed[train_paths[i]]
            for t in TRANSPOSITIONS:
                Xtr.append(task1_features(notes, tpq, tempos, ts, transpose=t))
                ytr.append(train_labels[i])
        Xtr = np.nan_to_num(np.array(Xtr))
        ytr = np.array(ytr)
        Xva = X_base[va_idx]
        yva = train_labels[va_idx]

        models = _task1_models()
        probs = []
        for name, m in models.items():
            m.fit(Xtr, ytr)
            pred = m.predict(Xva)
            fold_acc[name].append((pred == yva).mean())
            probs.append(m.predict_proba(Xva))
        ens = models["hgb"].classes_[np.argmax(np.mean(probs, axis=0), axis=1)]
        fold_acc["ensemble"].append((ens == yva).mean())

    for name, accs in fold_acc.items():
        print(f"[task1] {name}: honest CV acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # final fit on ALL training pieces (augmented across keys), predict the un-transposed test set
    Xtr, ytr = [], []
    for i, p in enumerate(train_paths):
        notes, tpq, tempos, ts = parsed[p]
        for t in TRANSPOSITIONS:
            Xtr.append(task1_features(notes, tpq, tempos, ts, transpose=t))
            ytr.append(train_labels[i])
    Xtr = np.nan_to_num(np.array(Xtr))
    ytr = np.array(ytr)
    X_test = np.nan_to_num(np.array([task1_features(*parsed[p]) for p in test_paths]))

    models = _task1_models()
    probs = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        probs.append(m.predict_proba(X_test))
    classes = models["hgb"].classes_
    preds = classes[np.argmax(np.mean(probs, axis=0), axis=1)]

    pred_dict = {p: int(preds[i]) for i, p in enumerate(test_paths)}
    with open("predictions1.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task1] wrote predictions1.json ({len(pred_dict)} entries)")


# Task 2: Temporal order prediction 

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


# Task 3: Audio tagging 

T3_DATAROOT = "student_files/task3_audio_classification"
SAMPLE_RATE = 22050
N_MELS = 96
N_CLASSES = 10
AUDIO_DURATION = 10
BATCH_SIZE = 64
T3_EPOCHS = 30
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance', 'blues', 'punk', 'chill', 'electronic', 'country']


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_waveform(path):
    waveform, sr = librosa.load(os.path.join(T3_DATAROOT, path), sr=SAMPLE_RATE)
    waveform = torch.FloatTensor(np.array([waveform]))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_len]
    return waveform


class AudioDataset(Dataset):
    def __init__(self, meta):
        self.meta = meta
        self.ids = list(meta.keys())
        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024,
                                  hop_length=256, f_min=20, f_max=SAMPLE_RATE // 2)
        self.db = AmplitudeToDB(top_db=80)
        # precompute mel spectrograms up front - the dataset is small enough to keep in memory
        self.feats = {}
        for path in tqdm(self.ids, desc="Preloading mels"):
            w = extract_waveform(path)
            m = self.db(self.mel(w)).squeeze(0)
            m = (m - m.mean()) / (m.std() + 1e-6)  # per-clip standardization
            self.feats[path] = m

    def spec_augment(self, m):
        # randomly zero out a frequency band and a time span (SpecAugment)
        if random.random() < 0.5:
            f = random.randint(0, 16)
            f0 = random.randint(0, max(1, N_MELS - f))
            m[f0:f0 + f, :] = 0
        if random.random() < 0.5:
            t = random.randint(0, 40)
            t0 = random.randint(0, max(1, m.shape[1] - t))
            m[:, t0:t0 + t] = 0
        return m

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = self.ids[idx]
        label = torch.tensor([1 if t in self.meta[path] else 0 for t in TAGS], dtype=torch.float32)
        return self.feats[path].unsqueeze(0), label, path


class SplitView(Dataset):
    # wraps a random_split subset so augmentation can be on for train and off for val
    def __init__(self, subset, augment):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path = self.dataset.ids[self.indices[idx]]
        label = torch.tensor([1 if t in self.dataset.meta[path] else 0 for t in TAGS], dtype=torch.float32)
        m = self.dataset.feats[path].clone()
        if self.augment:
            m = self.dataset.spec_augment(m)
        return m.unsqueeze(0), label, path


class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=(2, 4)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.block1 = conv_block(1, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)  # logits; sigmoid is applied later


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            y = y.to(device)
            all_logits.append(model(x).cpu())
            all_targets.append(y.cpu())
            all_paths += list(paths)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.sigmoid(logits).numpy()

    mAP = None
    if targets.sum() > 0:  # test set has no labels, so skip the metric there
        try:
            mAP = average_precision_score(targets, probs, average='macro')
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

    full_train = AudioDataset(train_meta)
    g = torch.Generator().manual_seed(0)
    n_train = int(len(full_train) * 0.9)
    n_valid = len(full_train) - n_train
    train_sub, valid_sub = random_split(full_train, [n_train, n_valid], generator=g)

    train_view = SplitView(train_sub, augment=True)
    valid_view = SplitView(valid_sub, augment=False)
    test_set = AudioDataset(test_meta)

    loader_train = DataLoader(train_view, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(valid_view, batch_size=BATCH_SIZE, shuffle=False)
    loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T3_EPOCHS)

    # keep the checkpoint with the best validation mAP
    best_map = -1.0
    best_state = None
    for epoch in range(T3_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y, _ in tqdm(loader_train, desc=f"Epoch {epoch+1}/{T3_EPOCHS}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
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
        key = p[2:] if p.startswith("./") else p  # autograder expects paths without the "./"
        pred_dict[key] = {TAGS[j]: float(probs[i][j]) for j in range(N_CLASSES)}

    with open("predictions3.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task3] wrote predictions3.json ({len(pred_dict)} entries)")


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
