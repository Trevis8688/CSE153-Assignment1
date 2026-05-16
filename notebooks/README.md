# Notebooks

Colab notebooks used for the heavier models. All four expect `student_files_updated.zip` to be present in Google Drive at `/content/drive/MyDrive/CSE153/student_files_updated.zip`. Each notebook mounts Drive, unzips the data into `/content/work/student_files/`, trains its model, and writes its predictions JSON back to Drive at the corresponding path.

Run them on a GPU runtime (Runtime, Change runtime type, GPU; A100 or V100 preferred).

## Files

- **`baseline_starter.ipynb`** — course-provided baseline notebook with simple working solutions for all three tasks. Mostly here as a reference for the expected data formats and output shapes.

- **`task1_remi_transformer.ipynb`** — Task 1 composer classification with a small BERT encoder trained from scratch over miditok REMI tokens. Augments by transposing in token space (-5 to +6 semitones). 6-layer encoder, 256-d hidden, 8 heads, around 5M parameters. Class-weighted cross-entropy, AdamW with cosine schedule, 30 epochs, 90/10 stratified split, multi-window logit averaging at inference. Writes `predictions1.json`.

- **`task3_cnn_from_scratch.ipynb`** — the original Task 3 4-block CNN over 96-bin log-mel spectrograms, SpecAugment, AdamW with cosine schedule, 30 epochs. Superseded by the CNN14 fine-tune below; kept for reference.

- **`task3_cnn14_finetune.ipynb`** — Task 3 audio tagging via fine-tuned PANNs CNN14 (pretrained on AudioSet at mAP 0.431). Discriminative learning rates (1e-4 backbone, 1e-3 head), mixup alpha 0.4, AdamW with cosine schedule, 25 epochs. This is the notebook that produced the current `predictions3.json`. Writes `predictions3.json`.

## Workflow

1. Train on Colab: run the notebook end to end, copy the resulting predictions JSON down from Drive (or via the file panel).
2. Place the file in the repository root, alongside the other `predictions*.json` files.
3. Update the matching section of `assignment1.py` and `writeup.txt` so they describe the model that produced the predictions.
4. Submit all five artifacts to Gradescope.
