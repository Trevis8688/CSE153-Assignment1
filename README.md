# CSE 153/253 Assignment 1

Code for the three-task music ML leaderboard assignment:

- **Task 1**: composer classification from MIDI (multi-class, accuracy)
- **Task 2**: temporal order prediction for MIDI segment pairs (binary, accuracy)
- **Task 3**: music tagging from 10-second audio clips (multi-label, mean average precision)

Each Gradescope submission requires five artifacts: `predictions1.json`, `predictions2.json`, `predictions3.json`, `assignment1.py`, and `writeup.txt`. The autograder evaluates the prediction files; the code and writeup are kept on record but not executed.

## Repository layout

```
.
├── assignment1.py                consolidated runnable code for all three tasks
├── writeup.txt                   short description of the current models, one paragraph per task
├── JOURNAL.md                    temporal log of every approach tried and its leaderboard result
├── 153 _ 253 2026 Assignment 1.md  course-provided assignment specification
├── notebooks/                    GPU notebooks (Colab) for the heavier models
│   ├── README.md                 short index of the notebooks
│   ├── baseline_starter.ipynb    course-provided baselines
│   ├── task1_remi_transformer.ipynb   from-scratch BERT encoder over REMI tokens
│   ├── task3_cnn_from_scratch.ipynb   the original 4-block CNN
│   └── task3_cnn14_finetune.ipynb     PANNs CNN14 fine-tuning, current Task 3 model
├── predictions1.json, predictions2.json, predictions3.json   submission outputs (gitignored)
└── student_files/                data, fetched from the course Drive (gitignored)
```

## Running

Local environment:

```
conda activate my-virtenv
python assignment1.py            # runs all three tasks, regenerates predictions*.json
```

Tasks 1 and 2 are fast on CPU. Task 3 (PANNs CNN14 fine-tuning) needs a GPU, so it is intended to be run from `notebooks/task3_cnn14_finetune.ipynb` on Colab Pro+. The notebook handles the Drive mount, data unzip, training, and writes `predictions3.json` back to Drive.

The MIDI Transformer for Task 1 lives in `notebooks/task1_remi_transformer.ipynb`. It is also run on Colab and writes `predictions1.json` to Drive.

## Data

The full data zip is not in the repo. Place `student_files_updated.zip` at `/content/drive/MyDrive/CSE153/student_files_updated.zip` for the Colab notebooks, or unzip it locally into `./student_files/` to run `assignment1.py` from the shell.

## Submission workflow

Every submission, even for a single-task change, uploads all five artifacts. Before submitting:

1. The three `predictions*.json` files are current.
2. `assignment1.py` reflects the models that actually produced those predictions.
3. `writeup.txt` describes the current models, not a stale earlier version.

Gradescope keeps the highest score per task across all submissions, so a regression on one task does not overwrite a previous best.
