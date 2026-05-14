"""Task 2: Temporal order prediction (binary).

For each pair (seg1, seg2): predict True if seg1 comes before seg2 in the
original piece, False otherwise. Build pairwise features comparing the
seg1->seg2 transition vs the seg2->seg1 transition, then train a classifier.
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

DATAROOT = "student_files/task2_next_sequence_prediction"


def _segment_summary(path, cache):
    if path in cache:
        return cache[path]
    m = miditoolkit.MidiFile(os.path.join(DATAROOT, path))
    tpq = m.ticks_per_beat or 480
    notes = []
    for inst in m.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    if not notes:
        out = {"empty": True}
        cache[path] = out
        return out

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    starts = np.array([n.start for n in notes], dtype=np.float64) / tpq
    ends = np.array([n.end for n in notes], dtype=np.float64) / tpq
    velocities = np.array([n.velocity for n in notes], dtype=np.float64)

    # Pitch class histogram
    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[int(p) % 12] += 1
    pc_hist /= pc_hist.sum() + 1e-9

    # First-K / Last-K boundary descriptors
    K = 5
    head_idx = np.argsort(starts)[:K]
    tail_idx = np.argsort(starts)[-K:]
    head_p = pitches[head_idx]
    tail_p = pitches[tail_idx]
    head_v = velocities[head_idx]
    tail_v = velocities[tail_idx]

    # The very first / very last note
    first_idx = int(np.argmin(starts))
    last_idx = int(np.argmax(starts))

    total_beats = float(ends.max() - starts.min())

    out = {
        "empty": False,
        "tpq": tpq,
        "num_notes": len(notes),
        "pitch_mean": float(pitches.mean()),
        "pitch_std": float(pitches.std()),
        "pitch_min": float(pitches.min()),
        "pitch_max": float(pitches.max()),
        "vel_mean": float(velocities.mean()),
        "vel_std": float(velocities.std()),
        "pc_hist": pc_hist,
        "head_pitch_mean": float(head_p.mean()),
        "head_pitch_std": float(head_p.std()),
        "tail_pitch_mean": float(tail_p.mean()),
        "tail_pitch_std": float(tail_p.std()),
        "head_vel_mean": float(head_v.mean()),
        "tail_vel_mean": float(tail_v.mean()),
        "first_pitch": float(pitches[first_idx]),
        "last_pitch": float(pitches[last_idx]),
        "first_vel": float(velocities[first_idx]),
        "last_vel": float(velocities[last_idx]),
        "total_beats": total_beats,
        "duration_last_note": float(ends[last_idx] - starts[last_idx]),
        "duration_first_note": float(ends[first_idx] - starts[first_idx]),
    }
    cache[path] = out
    return out


def _pair_features(s1, s2):
    """Features that describe the (s1, s2) ordering."""
    if s1.get("empty") or s2.get("empty"):
        return np.zeros(40, dtype=np.float64)

    # Transition: end of s1 to start of s2
    fwd_pitch_diff = s2["first_pitch"] - s1["last_pitch"]
    fwd_pitch_dist = abs(fwd_pitch_diff)
    fwd_tail_to_head = s2["head_pitch_mean"] - s1["tail_pitch_mean"]
    fwd_vel_diff = s2["first_vel"] - s1["last_vel"]
    # Pitch class continuity (cosine sim)
    a = s1["pc_hist"]; b = s2["pc_hist"]
    pc_sim_fwd = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Reverse transition: end of s2 to start of s1
    rev_pitch_diff = s1["first_pitch"] - s2["last_pitch"]
    rev_pitch_dist = abs(rev_pitch_diff)
    rev_tail_to_head = s1["head_pitch_mean"] - s2["tail_pitch_mean"]
    rev_vel_diff = s1["first_vel"] - s2["last_vel"]

    # Asymmetry: forward minus reverse (these are the discriminative ones)
    asym_pitch_dist = fwd_pitch_dist - rev_pitch_dist
    asym_tail_to_head = fwd_tail_to_head - rev_tail_to_head
    asym_vel = fwd_vel_diff - rev_vel_diff

    # Segment-level differences
    pitch_mean_diff = s1["pitch_mean"] - s2["pitch_mean"]
    pitch_std_diff = s1["pitch_std"] - s2["pitch_std"]
    vel_mean_diff = s1["vel_mean"] - s2["vel_mean"]
    note_count_diff = s1["num_notes"] - s2["num_notes"]
    duration_diff = s1["total_beats"] - s2["total_beats"]

    return np.array([
        fwd_pitch_diff, fwd_pitch_dist, fwd_tail_to_head, fwd_vel_diff, pc_sim_fwd,
        rev_pitch_diff, rev_pitch_dist, rev_tail_to_head, rev_vel_diff,
        asym_pitch_dist, asym_tail_to_head, asym_vel,
        pitch_mean_diff, pitch_std_diff, vel_mean_diff, note_count_diff, duration_diff,
        # Boundary pitches themselves (raw context)
        s1["last_pitch"], s2["first_pitch"], s1["first_pitch"], s2["last_pitch"],
        s1["tail_pitch_mean"], s2["head_pitch_mean"],
        s1["head_pitch_mean"], s2["tail_pitch_mean"],
        # Segment characteristics
        s1["pitch_mean"], s2["pitch_mean"],
        s1["pitch_std"], s2["pitch_std"],
        s1["vel_mean"], s2["vel_mean"],
        s1["num_notes"], s2["num_notes"],
        s1["total_beats"], s2["total_beats"],
        s1["duration_last_note"], s2["duration_first_note"],
        s1["last_vel"], s2["first_vel"],
        # PC sim repeated (symmetric anyway)
        pc_sim_fwd,
    ], dtype=np.float64)


def build_dataset(pairs, cache):
    feats = []
    for (p1, p2) in tqdm(pairs, desc="pairs"):
        s1 = _segment_summary(p1, cache)
        s2 = _segment_summary(p2, cache)
        feats.append(_pair_features(s1, s2))
    return np.array(feats, dtype=np.float64)


def train_predict_save():
    train = eval(open(os.path.join(DATAROOT, "train.json")).read())
    test = eval(open(os.path.join(DATAROOT, "test.json")).read())

    train_pairs = list(train.keys())
    train_labels = np.array([1 if train[k] else 0 for k in train_pairs])
    test_pairs = list(test)

    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    print(f"Label balance: {Counter(train_labels.tolist())}")

    cache = {}
    X_train = build_dataset(train_pairs, cache)
    X_test = build_dataset(test_pairs, cache)
    print(f"Feature dim: {X_train.shape[1]}")

    # Symmetry augmentation: swap pair => flip label
    swapped_pairs = [(p2, p1) for (p1, p2) in train_pairs]
    X_swap = build_dataset(swapped_pairs, cache)
    y_swap = 1 - train_labels

    X_all = np.concatenate([X_train, X_swap], axis=0)
    y_all = np.concatenate([train_labels, y_swap], axis=0)
    print(f"After augmentation: {X_all.shape}")

    X_all = np.nan_to_num(X_all, 0.0, 0.0, 0.0)
    X_test = np.nan_to_num(X_test, 0.0, 0.0, 0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=0),
        "gb": GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=0),
    }
    for name, m in candidates.items():
        s = cross_val_score(m, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  {name}: CV acc = {s.mean():.4f} +/- {s.std():.4f}")

    fitted = {}
    for name, m in candidates.items():
        m.fit(X_all, y_all)
        fitted[name] = m

    # Soft-vote ensemble
    probas = []
    for name, m in fitted.items():
        p = m.predict_proba(X_test)
        probas.append(p)
    avg_proba = np.mean(probas, axis=0)
    classes = fitted["gb"].classes_
    ens_idx = np.argmax(avg_proba, axis=1)
    ens_pred = classes[ens_idx]

    predictions = {tuple(test_pairs[i]): bool(ens_pred[i] == 1) for i in range(len(test_pairs))}

    out_path = "predictions2.json"
    with open(out_path, "w") as f:
        f.write(repr(predictions) + "\n")
    print(f"Wrote {out_path} ({len(predictions)} entries)")


if __name__ == "__main__":
    train_predict_save()
