"""Task 1: Composer classification.

Build a rich MIDI feature vector per file and train a gradient-boosting classifier.
"""
import os
import json
import numpy as np
import miditoolkit
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

DATAROOT = "student_files/task1_composer_classification"


def _safe_stats(arr):
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    a = np.asarray(arr, dtype=np.float64)
    return [float(a.mean()), float(a.std()), float(a.min()), float(a.max()), float(np.median(a))]


def extract_features(path):
    m = miditoolkit.MidiFile(os.path.join(DATAROOT, path))
    tpq = m.ticks_per_beat or 480

    # Pool notes across all instruments (most files have 1 inst but be safe)
    notes = []
    for inst in m.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if len(notes) == 0:
        # Fall back to whatever's there
        for inst in m.instruments:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))

    if len(notes) == 0:
        return np.zeros(80, dtype=np.float64)

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    durations_ticks = np.array([n.end - n.start for n in notes], dtype=np.float64)
    durations_beats = durations_ticks / tpq
    starts = np.array([n.start for n in notes], dtype=np.float64)
    velocities = np.array([n.velocity for n in notes], dtype=np.float64)

    # Inter-onset intervals (beats)
    iois = np.diff(np.sort(np.unique(starts))) / tpq
    # Melodic intervals on sorted-by-start notes
    intervals = np.diff(pitches[np.argsort(starts)])

    # Pitch class histogram (12 bins, normalized)
    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[int(p) % 12] += 1
    pc_hist /= pc_hist.sum() + 1e-9

    # Pitch register histogram (low/mid/high)
    reg_hist = np.zeros(8)
    for p in pitches:
        b = min(int(p) // 16, 7)
        reg_hist[b] += 1
    reg_hist /= reg_hist.sum() + 1e-9

    # Polyphony: count concurrent notes by sweeping events
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    poly = 0
    poly_samples = []
    for _, d in events:
        poly += d
        poly_samples.append(poly)
    poly_samples = np.array(poly_samples, dtype=np.float64) if poly_samples else np.array([0.0])

    # Tempo
    tempos = [t.tempo for t in m.tempo_changes] if m.tempo_changes else [120.0]
    tempos = np.array(tempos, dtype=np.float64)

    # Time signature numerator/denominator (use first)
    if m.time_signature_changes:
        ts_num = m.time_signature_changes[0].numerator
        ts_den = m.time_signature_changes[0].denominator
    else:
        ts_num, ts_den = 4, 4

    total_ticks = max((n.end for n in notes), default=1)
    total_beats = total_ticks / tpq
    note_density = len(notes) / max(total_beats, 1e-3)

    feats = []
    feats += _safe_stats(pitches)              # 5
    feats += _safe_stats(durations_beats)      # 5
    feats += _safe_stats(iois) if len(iois) else [0.0]*5  # 5
    feats += _safe_stats(intervals) if len(intervals) else [0.0]*5  # 5
    feats += _safe_stats(velocities)           # 5
    feats += _safe_stats(poly_samples)         # 5
    feats += list(pc_hist)                     # 12
    feats += list(reg_hist)                    # 8

    # Interval histogram (-12..+12 + outside)
    int_hist = np.zeros(27)
    for iv in intervals:
        b = int(np.clip(iv + 13, 0, 26))
        int_hist[b] += 1
    int_hist /= int_hist.sum() + 1e-9
    feats += list(int_hist)                    # 27

    # Duration histogram (in beats, quantized)
    dur_bins = [0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
    dur_hist = np.zeros(len(dur_bins) - 1)
    for d in durations_beats:
        for i in range(len(dur_bins) - 1):
            if dur_bins[i] <= d < dur_bins[i+1]:
                dur_hist[i] += 1
                break
    dur_hist /= dur_hist.sum() + 1e-9
    feats += list(dur_hist)                    # 8

    feats += [float(tempos.mean()), float(tempos.std()), float(tempos.min()), float(tempos.max())]  # 4
    feats += [float(ts_num), float(ts_den), float(note_density), float(len(notes)), float(total_beats)]  # 5

    return np.array(feats, dtype=np.float64)


def load_features(paths, cache=None, desc="extract"):
    feats = []
    for p in tqdm(paths, desc=desc):
        if cache is not None and p in cache:
            feats.append(cache[p])
        else:
            f = extract_features(p)
            if cache is not None:
                cache[p] = f
            feats.append(f)
    return np.array(feats)


def train_predict_save():
    train = eval(open(os.path.join(DATAROOT, "train.json")).read())
    test = eval(open(os.path.join(DATAROOT, "test.json")).read())

    train_paths = list(train.keys())
    train_labels = np.array([int(train[k]) for k in train_paths])
    test_paths = list(test)

    print(f"Train: {len(train_paths)} files, Test: {len(test_paths)} files")
    print(f"Label distribution: {Counter(train_labels.tolist())}")

    cache = {}
    X_train = load_features(train_paths, cache=cache, desc="train")
    X_test = load_features(test_paths, cache=cache, desc="test")
    print(f"Feature dim: {X_train.shape[1]}")

    # Replace any nan/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Quick model comparison via CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=0, class_weight="balanced"),
        "gb": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
    }
    cv_scores = {}
    for name, m in candidates.items():
        s = cross_val_score(m, X_train, train_labels, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_scores[name] = s.mean()
        print(f"  {name}: CV acc = {s.mean():.4f} +/- {s.std():.4f}")

    # Train all three, pick the best by CV; also create a soft-vote ensemble using probabilities
    best = max(cv_scores, key=cv_scores.get)
    print(f"Best single: {best}")

    # Fit all on full data
    fitted = {}
    for name, m in candidates.items():
        m.fit(X_train, train_labels)
        fitted[name] = m

    # Soft-vote ensemble of all models that have predict_proba
    probas = []
    for name, m in fitted.items():
        p = m.predict_proba(X_test)
        probas.append(p)
    avg_proba = np.mean(probas, axis=0)
    classes = fitted[best].classes_
    ensemble_preds = classes[np.argmax(avg_proba, axis=1)]

    # Use ensemble predictions
    predictions = {p: int(ensemble_preds[i]) for i, p in enumerate(test_paths)}

    out_path = "predictions1.json"
    with open(out_path, "w") as f:
        f.write(repr(predictions) + "\n")
    print(f"Wrote {out_path} ({len(predictions)} entries)")

    # Train accuracy of ensemble (sanity check)
    probas_tr = []
    for m in fitted.values():
        probas_tr.append(m.predict_proba(X_train))
    avg_tr = np.mean(probas_tr, axis=0)
    ens_tr = classes[np.argmax(avg_tr, axis=1)]
    print(f"Ensemble train accuracy: {(ens_tr == train_labels).mean():.4f}")


if __name__ == "__main__":
    train_predict_save()
