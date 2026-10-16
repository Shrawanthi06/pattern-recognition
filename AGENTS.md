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

# **Student Details**

* I'm Shrawanthi P (26120098), pursuing M.Tech AI and Data Science, who one day wants to become a MLE or AI Engineer. I have a course on Pattern Recognition and Machine Learning semester and this repo is all about that.

# **Coding & Notebook Rules**

* Don't Overcomplicate than what's necessary. For example : using pathlib to map the path when you have the data in the nearby directory and could have used './' notation.
* Use simple coding syntaxes in python, I'm a beginner python guy. So, need you to code reponsibily in a way that I can understand stuff.
* While using commands. Make the commands very concise and straight to the point and also use **word** (i.e.) I want every heading in bold format and don't use H1 pretty often. Use things like ### (i.e.) multiple hashes to make it small.
* The command should be very minimal as possible - Like use very few words.

# **Reports**

* Reports should be self-containted. When I ask for report, you don't write the content you only create the latex and put in the formulas, images and give me the template - Leaving out the content that needs to be filled.
