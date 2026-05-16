# Submission Journal

A running log of every approach attempted, the rationale behind it, and how it scored on the Gradescope leaderboard. Read it top to bottom for the full timeline. Each entry covers one submission.

Leaderboard format throughout: **Task 1 / Task 2 / Task 3** (rank when available).

## Submission 1 — baselines on top of hand features

**Date:** 2026-05-13
**Leaderboard:** 0.6633 / 0.8739 / 0.5958 (rank 120 of 278)

The first end-to-end run. Goal was to land a valid submission on the leaderboard before investing in any one task.

- **Task 1.** Each MIDI summarized as a 94-dimensional feature vector: pitch, duration, velocity, melodic interval, and inter-onset interval statistics, plus a pitch-class histogram, a register histogram, an interval histogram, a duration histogram, polyphony statistics, and tempo / time-signature / note-density information. A logistic regression, random forest, and gradient boosting classifier were trained on these features and their predicted probabilities were averaged (soft vote) to produce the final composer. Cross-validation showed about 0.90 accuracy.
- **Task 2.** Each segment was summarized (first and last note pitch and velocity, head and tail pitch means, pitch-class histogram, overall statistics). Pair features compared the segment-1-to-segment-2 transition against the segment-2-to-segment-1 transition. Because swapping a pair flips its label, the training set was doubled by including swapped versions. Same three-classifier soft-vote ensemble.
- **Task 3.** Each 10-second clip was loaded at 22.05 kHz and converted to a 96-bin log-mel spectrogram, standardized per clip. A 4-block convolutional network (Conv-BN-ReLU with max pooling, global average pooling, linear) trained with AdamW, cosine LR, BCE-with-logits for 30 epochs on Colab Pro+ GPU. SpecAugment-style frequency and time masking on training clips; 90/10 train/val split, best validation mAP checkpoint kept. Best val mAP about 0.557.

**What surprised us.** Task 1 cross-validation said 0.90 but the leaderboard returned 0.6633 — a 24-point gap. Task 2 and Task 3 tracked their validation numbers closely. The Task 1 gap dominated the rest of the iteration history.

## Submission 2 — Task 1: transposition augmentation, richer features, honest CV

**Date:** 2026-05-14
**Leaderboard:** 0.6131 / 0.8739 / 0.5958

Hypothesis: the Task 1 CV/leaderboard gap was a key-distribution shift, and augmenting across all twelve keys plus adding transposition-invariant features would close it.

- Same parsing pipeline, but pitches could be shifted by a semitone offset before features were computed, and each training MIDI was duplicated across offsets in `range(-5, 7)`.
- New features: a key-centered pitch-class histogram (rotated so the most common pitch class is bin zero), a skip-2 interval histogram, absolute interval statistics, an IOI-ratio histogram, and a chord-onset fraction. Total dimensionality 146.
- Cross-validation made honest: 5 stratified folds on the original pieces, augment only the training fold so a transposed copy of a piece never leaked into its own validation fold.
- Classifiers: logistic regression, `HistGradientBoostingClassifier`, random forest. Soft-vote ensemble.

Honest CV came in around 0.87 (vs. the inflated 0.90 from before). Leaderboard came back at 0.6131 — five points worse than baseline. The augmentation made things worse, not better.

**What this tells us.** Key-distribution shift was the wrong hypothesis. Augmenting across keys should have helped if it had been the bottleneck. A train/test feature-distribution diagnostic right after the regression showed only modest shifts — around 0.3 standard deviations at most for the largest-drifting features — so the CV/leaderboard gap is most likely overfitting to compound feature patterns that simply do not transfer between the train and test corpora.

Working assumption from here on: hand-feature work on Task 1 is at a ceiling. The bigger leaderboard ROI is on Task 3 (low baseline) and, if time permits, a learned sequence model for Task 1 that can discover its own invariances.

Reverted Task 1 in `assignment1.py` and `writeup.txt` back to the Submission 1 ensemble. `predictions1.json` regenerated from the reverted code.

## Submission 3 — Task 3: fine-tuned PANNs CNN14

**Date:** 2026-05-15
**Leaderboard:** 0.6633 / 0.8739 / 0.6450 (rank 147)

Pivoting to the place with the most headroom. Replaced the from-scratch CNN with a CNN14 backbone pretrained on AudioSet at mAP 0.431. The original 527-class classifier was discarded and the 2048-dimensional embedding produced by the backbone fed into a fresh two-layer head (linear, ReLU, dropout, linear) with ten outputs.

- Input audio fed directly as raw 32 kHz waveform (PANNs handles the log-mel front-end internally).
- Discriminative learning rates: 1e-4 for the pretrained backbone, 1e-3 for the new head.
- Mixup with alpha 0.4 on each training batch. AdamW with cosine schedule over 25 epochs.
- BCE-with-logits loss; 90/10 train/val split; best-checkpoint-by-val-mAP.

Best val mAP **0.6384** (up from 0.557 with the from-scratch CNN). Leaderboard moved to 0.6450 — a +0.0492 jump on Task 3 alone. Roughly tracked the validation metric within 0.01, as expected on this dataset.

Tasks 1 and 2 were unchanged this round. Notebook: `notebooks/task3_cnn14_finetune.ipynb`.

## Up next

- **Task 1: from-scratch REMI Transformer** (in flight). Notebook `notebooks/task1_remi_transformer.ipynb`. A small BERT encoder over miditok REMI tokens with transposition augmentation in token space and multi-window inference. Target: clear the 0.66 leaderboard ceiling that hand features hit.
- If Task 1 lands well, the same architecture can be reused for Task 2 (pair classification).
- Cheap potential Task 3 wins: test-time augmentation (average predictions over a few shifted crops) and seed-ensembling. Worth ~0.01-0.02 each, but only after Task 1 is dealt with.
