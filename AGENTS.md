# Repository Guidelines

## Project overview

This repository contains lab assignments for a Pattern Recognition and Machine
Learning course. It is a Python project managed with `uv` and requires Python
3.13 or newer.

## Repository structure

- `lab2/` contains the image reconstruction experiments using EVD and SVD.
- `lab3/` contains the polynomial regression and model-selection experiments.
- Each lab keeps source code in `src/`, input data in `input/`, and generated
  figures in `output/`.

## Setup and execution

Install the project dependencies from the repository root:

```bash
uv sync
```

Run Lab 2 from the repository root:

```bash
uv run python lab2/main.py
```

Run Lab 3 from its directory because its input and output paths are relative to
that directory:

```bash
cd lab3
uv run python main.py
```

## Change guidelines

- Keep changes focused on the requested lab or task.
- Do not commit virtual environments, Python cache files, or operating-system
  metadata.
- Preserve input datasets unless a task explicitly requires changing them.
- When changing an experiment, run the relevant lab and inspect its generated
  plots before committing.
- Check `git status` and review the diff before staging files.
- Do not overwrite unrelated work already present in the working tree.
