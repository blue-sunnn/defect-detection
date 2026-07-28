# Defect Detection Platform

Tkinter desktop application for copper-trace defect detection with three main workflows:
Dataset, Training, and Live. Training now owns evaluation, threshold and multiplier
review, promotion gating, restart recovery, and deployable package creation.

## Usage Notice

This project was created as an internship project for review and internal use.

The source code is provided for evaluation and authorized use only. Do not copy,
redistribute, sublicense, or use this software commercially without written
permission from the project owner.

## Important files

- `start_app.bat` — Windows launcher for users running the editable source app.
- `run_app.py` — run the application from source.
- `requirements.txt` — Python packages needed by the source app.
- `VERSION` — semantic application version. Edit this when releasing a new version.

## Organized folders

- `app/` — desktop UI.
- `backend/` — model and pipeline code.
- `assets/` — application logo.
- `config/` — local UI configuration.
- `docs/` — workflow notes.

## Workflow Handoff

For the ML-team workflow, artifact schemas, gate rules, restart behavior, promotion package
layout, Live package loading, and known integration gaps, see:

```text
docs/ML_TEAM_HANDOFF.md
```

## Run on Windows

Download the repository ZIP, unzip it, then double-click:

```bat
start_app.bat
```

The launcher creates a local `.venv`, installs packages from `requirements.txt`,
and starts the editable source application.

Python 3 must be installed on Windows before running the launcher. If Python is
missing, install it from python.org and enable "Add python.exe to PATH" during
installation.

## Run Manually

```bat
python run_app.py
```
