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

## Submission 4 — Task 1: from-scratch REMI Transformer

**Date:** 2026-05-15
**Best val acc:** 0.8926 (epoch 30)
**Leaderboard:** 0.5778 / 0.8739 / 0.6450 (rank 267)

Hand features had run out of room. Switched Task 1 to a learned sequence model so it could pick up its own transposition and corpus invariances instead of getting them hand-rolled.

- MIDIs tokenized with REMI via miditok (Bar, Position, Pitch, Velocity, Duration, Tempo, Time-signature tokens; vocab around 389).
- Model is a small BERT encoder trained from scratch: 6 layers, 256-dimensional hidden size, 8 heads, 1024-dimensional feed-forward, learned positional embeddings (about 5 million parameters). Sequences are 512-token windows with a BOS sentinel and right padding. Mean pooling over non-pad positions feeds a linear classifier head over the 8 composer classes.
- Transposition augmentation is implemented as a precomputed token-id remap per semitone offset in `range(-5, 7)`. At training time, each MIDI yields four random windows per epoch and each window is remapped with a randomly chosen offset. Bar, Position, Velocity, Duration, Tempo, and Time-signature tokens are untouched.
- Class-weighted cross-entropy (inverse frequency, normalized to mean 1) handles the 13x imbalance between classes 1 and 7.
- AdamW (lr 1e-4, weight decay 0.01), cosine schedule, grad-norm clip at 1.0, 30 epochs.
- Validation split: stratified 90/10 over the original pieces. The val fold is never transposed and never windowed in training. Multi-window logit averaging is used both at validation time and at test time.

Best val accuracy reached 0.8926 by epoch 30, but the leaderboard came back at 0.5778 - a 31-point validation-to-leaderboard gap, worse than the 24-point gap the hand-feature ensemble had. The Transformer overfit *more* than the hand features did, not less. Notebook: `notebooks/task1_remi_transformer.ipynb`.

The likeliest single contributor is the inverse-frequency class weighting. Training labels are distributed 17 / 40 / 11 / 10 / 4 / 10 / 4 / 3 percent across the eight composers, but the Transformer's predictions on the test set came out 29 / 26 / 12 / 13 / 7 / 6 / 4 / 3 percent: predictions were pulled sharply away from the dominant class 1. If the leaderboard test set has class proportions close to the training set, that rebalancing actively hurts.

## Submission 5 — Revert Task 1 to the 94-feature baseline

**Date:** 2026-05-16
**Leaderboard:** 0.6633 / 0.8739 / 0.6450 (best-of-submissions, since Gradescope keeps the highest per-task score)

Task 1 reverted: `assignment1.py` Task 1 section restored to the 94-feature logistic regression / random forest / gradient boosting soft-vote ensemble that produced Submission 1, `writeup.txt` Task 1 paragraph restored, `predictions1.json` regenerated from the reverted code (deterministic with seed 0). The Transformer experiment was a clear regression and there is no cheap fix in sight: a third Task 1 experiment in a row that underperformed the very first baseline.

Gradescope keeps the highest score per task across submissions, so the rank-relevant Task 1 score is still 0.6633 from Submission 1. Treating Task 1 as closed for now.

## Up next

- **Task 3 follow-ups (low risk, low reward, around +0.02 mAP).** Test-time augmentation (average predictions over a few shifted crops) and a second-seed ensemble. Cheap to wire up on Colab.
- **Task 2 sequence model (higher risk, higher reward).** The same small Transformer backbone as Task 1 but for pair classification: concatenate `[BOS] tok(seg1) [SEP] tok(seg2)`, encode, binary head. Symmetric augmentation by swapping pair order at training time. Caveat: Task 1 told us a small Transformer can overfit hard on this dataset, so the same risk applies.
- **No more hand-feature Task 1 work.** Three attempts have all underperformed the baseline.
