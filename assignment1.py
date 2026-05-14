"""Assignment 1 solution: composer classification, temporal-order prediction, and
audio tagging.

Running this file end-to-end reproduces ``predictions1.json``,
``predictions2.json``, and ``predictions3.json``. Each task is self-contained;
the file is organised as three sections plus a small main.

Author: Trevor Duong (CSE 153 / 253, 2026 — Assignment 1).
"""
from __future__ import annotations

import os
import random
from collections import Counter

import librosa
import miditoolkit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Task 1: Composer classification
# ---------------------------------------------------------------------------

T1_DATAROOT = "student_files/task1_composer_classification"


def _safe_stats(arr):
    if len(arr) == 0:
        return [0.0] * 5
    a = np.asarray(arr, dtype=np.float64)
    return [float(a.mean()), float(a.std()), float(a.min()), float(a.max()), float(np.median(a))]


def t1_features(path):
    m = miditoolkit.MidiFile(os.path.join(T1_DATAROOT, path))
    tpq = m.ticks_per_beat or 480
    notes = []
    for inst in m.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        for inst in m.instruments:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    if not notes:
        return np.zeros(94)

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    durs = np.array([n.end - n.start for n in notes], dtype=np.float64) / tpq
    starts = np.array([n.start for n in notes], dtype=np.float64)
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    iois = np.diff(np.sort(np.unique(starts))) / tpq
    intervals = np.diff(pitches[np.argsort(starts)])

    pc = np.zeros(12)
    for p in pitches:
        pc[int(p) % 12] += 1
    pc /= pc.sum() + 1e-9

    reg = np.zeros(8)
    for p in pitches:
        reg[min(int(p) // 16, 7)] += 1
    reg /= reg.sum() + 1e-9

    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    poly = 0
    samp = []
    for _, d in events:
        poly += d
        samp.append(poly)
    samp = np.array(samp or [0.0], dtype=np.float64)

    tempos = np.array([t.tempo for t in m.tempo_changes] or [120.0], dtype=np.float64)
    if m.time_signature_changes:
        tsn, tsd = m.time_signature_changes[0].numerator, m.time_signature_changes[0].denominator
    else:
        tsn, tsd = 4, 4

    total_ticks = max((n.end for n in notes), default=1)
    total_beats = total_ticks / tpq
    density = len(notes) / max(total_beats, 1e-3)

    feats = []
    feats += _safe_stats(pitches)
    feats += _safe_stats(durs)
    feats += _safe_stats(iois) if len(iois) else [0.0] * 5
    feats += _safe_stats(intervals) if len(intervals) else [0.0] * 5
    feats += _safe_stats(vels)
    feats += _safe_stats(samp)
    feats += list(pc)
    feats += list(reg)

    ih = np.zeros(27)
    for iv in intervals:
        ih[int(np.clip(iv + 13, 0, 26))] += 1
    ih /= ih.sum() + 1e-9
    feats += list(ih)

    db = [0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
    dh = np.zeros(len(db) - 1)
    for d in durs:
        for i in range(len(db) - 1):
            if db[i] <= d < db[i + 1]:
                dh[i] += 1
                break
    dh /= dh.sum() + 1e-9
    feats += list(dh)

    feats += [float(tempos.mean()), float(tempos.std()), float(tempos.min()), float(tempos.max())]
    feats += [float(tsn), float(tsd), float(density), float(len(notes)), float(total_beats)]
    return np.array(feats, dtype=np.float64)


def run_task1():
    train = eval(open(os.path.join(T1_DATAROOT, "train.json")).read())
    test = eval(open(os.path.join(T1_DATAROOT, "test.json")).read())

    train_paths = list(train.keys())
    train_labels = np.array([int(train[k]) for k in train_paths])
    test_paths = list(test)

    cache = {}
    def load(paths, desc):
        out = []
        for p in tqdm(paths, desc=desc):
            if p not in cache:
                cache[p] = t1_features(p)
            out.append(cache[p])
        return np.nan_to_num(np.array(out), 0.0, 0.0, 0.0)

    X_train = load(train_paths, "task1 train")
    X_test = load(test_paths, "task1 test")

    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0, class_weight="balanced"),
        "gb": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, m in candidates.items():
        s = cross_val_score(m, X_train, train_labels, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"[task1] {name}: CV acc = {s.mean():.4f} +/- {s.std():.4f}")

    fitted = {n: m.fit(X_train, train_labels) for n, m in candidates.items()}
    probs = np.mean([m.predict_proba(X_test) for m in fitted.values()], axis=0)
    classes = fitted["gb"].classes_
    preds = classes[np.argmax(probs, axis=1)]
    pred_dict = {p: int(preds[i]) for i, p in enumerate(test_paths)}
    with open("predictions1.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task1] wrote predictions1.json ({len(pred_dict)} entries)")


# ---------------------------------------------------------------------------
# Task 2: Temporal order
# ---------------------------------------------------------------------------

T2_DATAROOT = "student_files/task2_next_sequence_prediction"


def _segment_summary(path, cache):
    if path in cache:
        return cache[path]
    m = miditoolkit.MidiFile(os.path.join(T2_DATAROOT, path))
    tpq = m.ticks_per_beat or 480
    notes = []
    for inst in m.instruments:
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

    K = 5
    head_idx = np.argsort(starts)[:K]
    tail_idx = np.argsort(starts)[-K:]
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


def _pair_features(s1, s2):
    if s1.get("empty") or s2.get("empty"):
        return np.zeros(40, dtype=np.float64)
    fwd_pd = s2["first_pitch"] - s1["last_pitch"]
    rev_pd = s1["first_pitch"] - s2["last_pitch"]
    fwd_th = s2["head_pitch_mean"] - s1["tail_pitch_mean"]
    rev_th = s1["head_pitch_mean"] - s2["tail_pitch_mean"]
    fwd_v = s2["first_vel"] - s1["last_vel"]
    rev_v = s1["first_vel"] - s2["last_vel"]
    a = s1["pc_hist"]; b = s2["pc_hist"]
    pc_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return np.array([
        fwd_pd, abs(fwd_pd), fwd_th, fwd_v, pc_sim,
        rev_pd, abs(rev_pd), rev_th, rev_v,
        abs(fwd_pd) - abs(rev_pd), fwd_th - rev_th, fwd_v - rev_v,
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
        return np.array([_pair_features(_segment_summary(p1, cache), _segment_summary(p2, cache))
                         for (p1, p2) in tqdm(pairs, desc=desc)])

    X_train = build(train_pairs, "task2 train")
    X_test = build(test_pairs, "task2 test")
    # symmetric augmentation: swap pair, flip label
    X_swap = build([(b, a) for (a, b) in train_pairs], "task2 swapped")
    X_all = np.nan_to_num(np.concatenate([X_train, X_swap]), 0.0, 0.0, 0.0)
    y_all = np.concatenate([train_labels, 1 - train_labels])
    X_test = np.nan_to_num(X_test, 0.0, 0.0, 0.0)

    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
        "gb": GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=0),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, m in candidates.items():
        s = cross_val_score(m, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"[task2] {name}: CV acc = {s.mean():.4f} +/- {s.std():.4f}")

    fitted = {n: m.fit(X_all, y_all) for n, m in candidates.items()}
    probs = np.mean([m.predict_proba(X_test) for m in fitted.values()], axis=0)
    classes = fitted["gb"].classes_
    preds = classes[np.argmax(probs, axis=1)]
    pred_dict = {tuple(test_pairs[i]): bool(preds[i] == 1) for i in range(len(test_pairs))}
    with open("predictions2.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task2] wrote predictions2.json ({len(pred_dict)} entries)")


# ---------------------------------------------------------------------------
# Task 3: Audio tagging
# ---------------------------------------------------------------------------

T3_DATAROOT = "student_files/task3_audio_classification"
SAMPLE_RATE = 22050
N_MELS = 96
N_CLASSES = 10
AUDIO_DURATION = 10
BATCH_SIZE = 64
T3_EPOCHS = 30
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance', 'blues', 'punk', 'chill', 'electronic', 'country']


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_waveform(path):
    waveform, sr = librosa.load(os.path.join(T3_DATAROOT, path), sr=SAMPLE_RATE)
    waveform = torch.FloatTensor(np.array([waveform]))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    tlen = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < tlen:
        waveform = F.pad(waveform, (0, tlen - waveform.shape[1]))
    else:
        waveform = waveform[:, :tlen]
    return waveform


class _AudioDataset(Dataset):
    def __init__(self, meta):
        self.meta = meta
        self.ids = list(meta.keys())
        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256, f_min=20, f_max=SAMPLE_RATE // 2)
        self.db = AmplitudeToDB(top_db=80)
        self.feats = {}
        for path in tqdm(self.ids, desc="Preloading mels"):
            w = _extract_waveform(path)
            m = self.db(self.mel(w)).squeeze(0)
            m = (m - m.mean()) / (m.std() + 1e-6)
            self.feats[path] = m

    def _augment(self, m):
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
        bin_label = torch.tensor([1 if t in self.meta[path] else 0 for t in TAGS], dtype=torch.float32)
        return self.feats[path].unsqueeze(0), bin_label, path


class _AudSubset(Dataset):
    def __init__(self, subset, augment):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = self.dataset.ids[real_idx]
        bin_label = torch.tensor([1 if t in self.dataset.meta[path] else 0 for t in TAGS], dtype=torch.float32)
        m = self.dataset.feats[path].clone()
        if self.augment:
            m = self.dataset._augment(m)
        return m.unsqueeze(0), bin_label, path


class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ic, oc, pool=(2, 4)):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )
        self.b1 = block(1, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


def _evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, ps in loader:
            x = x.to(device); y = y.to(device)
            all_logits.append(model(x).cpu())
            all_targets.append(y.cpu())
            all_paths += list(ps)
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits).numpy(); targets_np = targets.numpy()
    mAP = None
    if targets_np.sum() > 0:
        try:
            mAP = average_precision_score(targets_np, probs, average='macro')
        except Exception:
            mAP = None
    return probs, all_paths, mAP


def run_task3():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = _device(); print("[task3] device:", device)

    train_meta = eval(open(os.path.join(T3_DATAROOT, "train.json")).read())
    test_meta = {k: [] for k in eval(open(os.path.join(T3_DATAROOT, "test.json")).read())}

    all_train = _AudioDataset(train_meta)
    g = torch.Generator().manual_seed(0)
    n = len(all_train); n_tr = int(n * 0.9); n_va = n - n_tr
    tr_sub, va_sub = random_split(all_train, [n_tr, n_va], generator=g)
    tr = _AudSubset(tr_sub, augment=True)
    va = _AudSubset(va_sub, augment=False)
    te = _AudioDataset(test_meta)

    loader_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_va = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False)
    loader_te = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    epochs = T3_EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_map = -1.0; best_state = None
    for ep in range(epochs):
        model.train(); running = 0.0; nb = 0
        for x, y, _ in tqdm(loader_tr, desc=f"Epoch {ep+1}/{epochs}"):
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        sched.step()
        _, _, vmap = _evaluate(model, loader_va, device)
        print(f"[task3 ep{ep+1}] loss={running/nb:.4f} val_mAP={vmap:.4f}")
        if vmap is not None and vmap > best_map:
            best_map = vmap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[task3] best val mAP: {best_map:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    probs, paths, _ = _evaluate(model, loader_te, device)
    pred_dict = {(p[2:] if p.startswith("./") else p):
                 {TAGS[j]: float(probs[i][j]) for j in range(N_CLASSES)}
                 for i, p in enumerate(paths)}
    with open("predictions3.json", "w") as f:
        f.write(repr(pred_dict) + "\n")
    print(f"[task3] wrote predictions3.json ({len(pred_dict)} entries)")


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
