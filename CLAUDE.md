# CSE 153/253 Assignment 1 — Project Context

## What this is
A 3-task music ML assignment graded on a Gradescope leaderboard.

- **Task 1**: composer classification from MIDI (multi-class, accuracy)
- **Task 2**: temporal order prediction for MIDI segment pairs (binary, accuracy)
- **Task 3**: music tagging from 10-second audio clips (multi-label, mean average precision)

Leaderboard scores have three numbers in the order Task 1 / Task 2 / Task 3.

## Submission workflow (IMPORTANT)
**Every single submission requires uploading all five artifacts**, even if only one task changed:

1. `predictions1.json`
2. `predictions2.json`
3. `predictions3.json`
4. `assignment1.py` — must be the consolidated, runnable code for all three tasks
5. `writeup.txt` — short description of the current approach for each task

So before any submission:
- Confirm `predictions1/2/3.json` exist and are current
- `assignment1.py` reflects whatever models actually produced those predictions
- `writeup.txt` describes the current models (not a stale earlier version)

If only one task changed, the other two tasks' code + writeup sections still need to be present and accurate.

## Working setup
- Conda env: `my-virtenv` (Python 3.10, torch on MPS for local). Activate with `conda activate my-virtenv`.
- Heavy training (Task 3, any neural model) runs on **Colab Pro+** (GPU). User uploads data zip to `/content/drive/MyDrive/CSE153/student_files_updated.zip`; notebook lives at `task3_colab.ipynb` (use it as the template for new Colab notebooks).
- GitHub remote: https://github.com/Trevis8688/CSE153-Assignment1 (branch `main`).

## Repo conventions
- **`.claude/` MUST stay in `.gitignore`** (user-stated security requirement — do not remove).
- `student_files/`, `student_files_updated.zip`, `predictions*.json`, `*.log` are gitignored — they're regenerated or fetched from Drive.
- Data files are not in the repo; assume `student_files/` is populated locally when running.

## Code style preferences (user-stated)
- Human-looking code: no formal module docstrings, no `from __future__ import annotations`, no unused imports.
- Expand abbreviations when reasonable (e.g. write things out rather than cryptic short names).
- No first-person in `writeup.txt` (no "I trained...", "we used...").
- Minimal comments — only when the *why* is non-obvious.

## Scoring history (for context)
First leaderboard submission: Task 1 = 0.6633, Task 2 = 0.8739, Task 3 = 0.5958 (rank 120/278).

Task 1 had a big CV/leaderboard gap (CV ~0.90 → leaderboard 0.66) caused by key-distribution shift; fixed in `task1_v2.py` with transposition augmentation + honest CV (augment only the training fold).
