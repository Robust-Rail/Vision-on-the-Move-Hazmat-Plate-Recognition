# Code Conventions: UN-Number Detection Project

This document outlines the essential coding standards, project structure, and workflow conventions for the `UN-NUMBER-DETECTION` project. Our goal is to maintain a **clean, efficient Git repository** that's easy to collaborate on, while minimizing unnecessary overhead.

**The core principle: If it's big or can be easily regenerated, it doesn't belong in Git.**

## Table of Contents

1.  [Project Structure (The Essentials)](#1-project-structure-the-essentials)
2.  [Tooling & Automation (Work Smarter, Not Harder)](#2-tooling--automation-work-smarter-not-harder)

    - [Linters & Formatters](#linters--formatters)
    - [Jupyter Notebook Output Stripping (`nbdev`)](#jupyter-notebook-output-stripping-nbdev)

3.  [Coding Style Guidelines (PEP 8 & Core Readability)](#3-coding-style-guidelines-pep-8--core-readability)
    - [Naming Conventions](#naming-conventions)
    - [Line Length](#line-length)
    - [Imports](#imports)
    - [Whitespace](#whitespace)
    - [Comments ](#comments)
4.  [Jupyter Notebook Workflow (For Exploration, Not Storage)](#4-jupyter-notebook-workflow-for-exploration-not-storage)
    - [Purpose of Notebooks](#purpose-of-notebooks)
    - [Refactoring Code from Notebooks](#refactoring-code-from-notebooks)
5.  [General Project Workflow (Streamlined)](#5-general-project-workflow-streamlined)
6.  [Git & Version Control (Keep it Clean)](#6-git--version-control-keep-it-clean)

---

## 1. Project Structure

This streamlined structure focuses on what absolutely needs to be in Git. Large data and generated outputs are explicitly kept out. Each folder and file's purpose is described below.

```
UN-NUMBER-DETECTION/
├── .git/                         # Git's internal directory; tracks repository history.
├── .gitattributes                # Defines Git behaviors, including nbdev filter for notebooks.
├── .gitignore                    # Specifies intentionally untracked files and directories (crucial for excluding large data/outputs).
├── .venv/                        # Python virtual environment; isolated dependencies for the project (always gitignored).
├── .pre-commit-config.yaml       # Configuration file for the pre-commit hook framework.
├── configs/                      # Stores configuration files for models, training, and data paths.
│   ├── base_config.yaml          # General, shared project configurations and default settings.
│   ├── faster_rcnn_config.yaml   # Specific configuration parameters for the Faster R-CNN model.
│   ├── yolo_config.yaml          # Specific configuration parameters for the YOLO model.
│   └── ocr_config.yaml           # Specific configuration parameters for OCR models.
├── data/                         # **All data files (THIS DIRECTORY IS GITIGNORED)**; contains raw and processed datasets.
│   ├── raw/                      # Original, immutable datasets as initially received.
│   │   ├── prorail_dataset/      # Raw image/data files specific to the Prorail dataset.
│   │   └── haztruck_dataset/     # Raw image/data files specific to the Haztruck dataset.
│   ├── processed/                # Datasets after cleaning, transformation, and splitting.
│   │   ├── prorail/              # Processed data for Prorail, formatted for model training (e.g., videos and csv).
│   │   └── haztruck/ # Processed data for Haztruck, formatted for model training.
│   └── annotations/              # Stores annotation files that may be separate from images (e.g., large COCO JSONs).
│       ├── haztruck_annotations_yolo/ # YOLO-format annotations (one .txt per image)
│       └── haztruck_annotations_coco.json # COCO-format annotations
├── notebooks/                    # Jupyter notebooks for interactive exploration, visualization, and reporting.
│   ├── 01_data_exploration.ipynb # Notebook for initial data loading, statistics, and sanity checks.
│   ├── 02_data_preprocessing_prorail.ipynb # Interactive steps for preprocessing the Prorail dataset.
│   ├── 03_data_preprocessing_haztruck.ipynb # Interactive steps for preprocessing the Haztruck dataset.
│   ├── 04_data_augmentation_experiments.ipynb # Notebooks for experimenting with various data augmentation techniques.
│   ├── 05_train_faster_rcnn.ipynb # Interactive notebook for fine-tuning or experimenting with Faster R-CNN training.
│   ├── 06_train_yolo.ipynb       # Interactive notebook for fine-tuning or experimenting with YOLO training.
│   ├── 07_evaluate_faster_rcnn.ipynb # Notebook to evaluate the Faster R-CNN model performance.
│   ├── 08_evaluate_yolo.ipynb    # Notebook to evaluate the YOLO model performance.
│   ├── 09_evaluate_ocr_models.ipynb # Notebook to evaluate various OCR model performances.
│   ├── 10_full_pipeline_analysis.ipynb # Notebook for holistic analysis of the integrated detection + OCR pipeline.
│   └── archive/                  # Contains older, deprecated, or experimental notebooks not actively used.
├── outputs/                      # **Generated outputs (THIS DIRECTORY IS GITIGNORED)**; stores results, models, and predictions.
│   ├── models/                   # Stores trained model weights and checkpoints.
│   │   ├── faster_rcnn/          # Saved model states for the Faster R-CNN detector.
│   │   │   └── best_model.pth    # The best performing Faster R-CNN model checkpoint.
│   │   ├── yolo/                 # Saved model states for the YOLO detector.
│   │   │   └── best_model.pt     # The best performing YOLO model checkpoint.
│   │   └── ocr/                  # Saved model states for OCR models.
│   │       └── best_model.pt     # The best performing OCR model checkpoint.
│   ├── results/                  # Stores evaluation metrics, logs, and plots.
│   │   ├── evaluation_faster_rcnn.json # JSON file containing detailed Faster R-CNN evaluation results.
│   │   ├── evaluation_yolo.json  # JSON file containing detailed YOLO evaluation results.
│   │   ├── evaluation_ocr.csv    # CSV file containing detailed OCR evaluation results.
│   │   └── plots/                # Generated plots and figures from analyses.
│   └── predictions/              # Stores example images with predictions (e.g., bounding boxes, detected text).
├── scripts/                      # Standalone Python scripts for repeatable, non-interactive tasks.
│   ├── prepare_data.py           # Script to run the entire data preprocessing pipeline.
│   ├── annotate_data.py          # Wrapper script to launch annotation logic using `annotation/cli`.
│   ├── train.py                  # Main script to initiate model training (e.g., `python train.py --model yolo`).
│   ├── evaluate.py               # Main script to run model evaluations and generate reports.
│   └── run_pipeline.py           # Script to execute the full detection + OCR pipeline on new data.
├── src/un_detector/              # Core source code for the project, structured as a Python package.
│   ├── __init__.py               # Marks `un_detector` as a Python package, allowing imports.
│   ├── data/                     # Modules for data loading, processing, and augmentation.
│   │   ├── __init__.py           # Marks `data` as a sub-package.
│   │   ├── datasets.py           # Defines PyTorch Dataset classes for loading specific data formats.
│   │   ├── augmentation.py       # Functions and classes for applying data augmentation techniques.
│   │   └── preprocessing.py      # Functions for cleaning, transforming, and preparing raw data.
│   ├── models/                   # Model definitions, architectures, and wrappers for different detector/OCR types.
│   │   ├── __init__.py           # Marks `models` as a sub-package.
│   │   ├── faster_rcnn_detector.py # Implements or wraps the Faster R-CNN model logic.
│   │   ├── yolo_detector.py        # Implements or wraps the YOLO model logic.
│   │   └── ocr_reader.py           # Wrapper for integrating and using OCR models (e.g., Idefics2).
│   ├── pipeline/                 # Code to orchestrate the full detection & OCR workflow.
│   │   ├── __init__.py           # Marks `pipeline` as a sub-package.
│   │   └── main_pipeline.py      # Contains the main logic for running the end-to-end detection and OCR pipeline.
│   ├── training/                 # Modules related to model training loops and utilities.
│   │   ├── __init__.py           # Marks `training` as a sub-package.
│   │   ├── train_utils.py        # Generic utilities for training (e.g., logging, checkpointing, early stopping).
│   │   └── trainers.py           # Specific trainer classes for different models (e.g., `FasterRCNNTrainer`).
│   ├── evaluation/               # Modules for evaluating model performance.
│   │   ├── __init__.py           # Marks `evaluation` as a sub-package.
│   │   └── metrics.py            # Implementations of various evaluation metrics (e.g., mAP, accuracy).
│   └── utils/                    # General utility functions used across the project.
│       ├── __init__.py           # Marks `utils` as a sub-package.
│       ├── config_utils.py       # Utilities for loading and parsing configuration files (YAML).
│       ├── logging_utils.py      # Standardized logging setup and helper functions.
│       └── vis_utils.py          # Utilities for creating plots and visualizations.
│   ├── annotation/                   # Dedicated module for handling annotation logic (multiple sources/formats)
│       ├── __init__.py               # Makes `annotation` a Python package.
│       ├── image_annotator.py        # Functions/classes for annotating static image datasets.
│       ├── video_annotator.py        # Extracts frames from videos and prepares them for annotation.
│       ├── utils.py                  # Shared utilities for annotation (e.g., filename mapping, ID correction).
│       ├── converters/               # Modules for converting raw annotations into specific formats.
│   │       ├── __init__.py           # Marks `converters` as a sub-package.
│   │       ├── coco_converter.py     # Converts annotations into COCO format (JSON).
│   │       ├── yolo_converter.py     # Converts annotations into YOLO format (TXT).
│   │       └── labelstudio_parser.py # Parses annotations exported from Label Studio (optional).
│   └── cli/                      # CLI interface to run annotation pipelines.
│       └── generate_annotations.py # Scriptable entry point to orchestrate annotation workflow from CLI.
├── .env                          # Local environment variables (e.g., API keys); **this file is gitignored**.
├── .env.example                  # Example file for `.env`, showing required environment variables (committed for reference).
├── README.md                     # Main project overview, setup instructions, and usage guide (essential for onboarding).
└── requirements.txt              # Lists all Python package dependencies required for the project.
```

**What we're omitting from Git (and how we handle it):**

- **`data/`:** **This is explicitly gitignored.** All raw and processed datasets are too large for Git. We will share these via a dedicated cloud storage solution (e.g., university shared drive, Google Drive) and provide clear download instructions in the `README.md`.
- **`outputs/`:** **This is explicitly gitignored.** All trained models, evaluation results, and generated plots are large and can be regenerated from code and data. These should also be managed via external storage or simply regenerated locally as needed.
- **`docs/`:** **This entire folder is OPTIONAL.** If you don't need extensive documentation beyond the `README.md` (e.g., formal design documents, API references), you can omit this directory.
- **`LICENSE`:** **This file is OPTIONAL.** For small, internal, or university projects, a formal license might not be necessary unless specifically required.
- **Virtual Environment (`.venv/`):** Already handled by `.gitignore`. You create it locally, but it's never committed.

---

## 2. Tooling & Automation (Work Smarter, Not Harder)

These tools automate tedious tasks, making it easier to keep the repository clean and consistent without manual effort.

### Linters & Formatters

Set these up once, and they'll handle formatting automatically.

- **Black:** An uncompromising code formatter. Just run it, and your code is formatted consistently.
- **Flake8:** Catches common errors and enforces style guidelines.
- **isort:** Automatically sorts imports.

**Setup and Usage:**

1.  **Installation:**
    ```bash
    pip install black flake8 isort
    ```
2.  **Integration:** Set up your IDE (VS Code, PyCharm) to run Black on save. Run Flake8 and isort manually or as pre-commit hooks (optional, but good for "lazy" consistency).

### Jupyter Notebook Output Stripping (`nbdev`)

This is **ESSENTIAL** for keeping notebooks lightweight in Git. It automatically removes all cell outputs (including large images) from `.ipynb` files.

**How it handles large notebooks and prevents "LFS territory":**
The core problem with notebooks in Git is not just their total file size, but the embedded outputs (like base64 encoded images or large text outputs) that make Git diffs unreadable and history bloated. `nbdev` acts as a Git `clean` filter. This means that when you `git add` an `.ipynb` file, `nbdev` **automatically processes and strips all outputs from it first**. The version of the notebook that is then staged in Git's index (ready for commit) is already clean and significantly smaller, containing only the code and markdown. This ensures that notebooks themselves never grow into "LFS territory" due to their outputs, and their diffs remain clean.

**Setup:**

1.  **Installation:**
    ```bash
    pip install nbdev
    ```
2.  **Configure Git (from project root):**
    ```bash
    nbdev --install
    ```
    This adds a Git filter so outputs are automatically stripped on `git add`.
3.  **Commit `.gitattributes`:**
    `bash
git add .gitattributes
git commit -m "Configure nbdev for Jupyter notebooks"
`
    **All team members must run `nbdev --install` once per clone** to ensure consistency.

## 3. Coding Style Guidelines

We adhere to the core principles of [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Black will enforce most of these automatically.

### Naming Conventions

- **Variables, Functions, Methods, Modules:** `lowercase_with_underscores`.
- **Constants:** `UPPERCASE_WITH_UNDERSCORES`.
- **Classes:** `PascalCase`.

### Line Length

- **Follow Black:** Black will handle line length wrapping. Generally aims for around 100 characters.

### Imports

- **One Import Per Line.**
- **Grouping Order:** `isort` will handle this automatically:
  1.  Standard library imports
  2.  Third-party library imports
  3.  Local application/library specific imports
      _(Run `isort` regularly or integrate into IDE.)_

### Whitespace

- **Follow Black:** Black will automatically ensure consistent spacing around operators and commas.

### Comments

- **Comments:** Use sparingly to clarify non-obvious code.

---

## 4. Jupyter Notebook Workflow (For Exploration, Not Storage)

Notebooks are for interactive work, not for storing large data or production code.

### Purpose of Notebooks

- **Quick Exploration:** Data analysis, prototyping new ideas.
- **Visualization:** Generating temporary plots (which are then stripped by `nbdev`).
- **Analysis & Reporting:** Presenting results and findings in a narrative format.

### Refactoring Code from Notebooks

- **Move Reusable Code to `src/`:** Any functions or classes that you find yourself reusing, or that represent core logic (data preprocessing, model definitions, evaluation), **must** be moved from notebooks into `.py` files within the `src/un_detector/` package.
- **Import in Notebooks:** Once moved, import these functions/classes into your notebooks.

  ```python
  import sys
  sys.path.append('../src') # If notebooks are not running from the root

  from un_detector.data.preprocessing import clean_text
  from un_detector.models.yolo_detector import YOLOModel
  # ... use them here ...
  ```

- **Handle Large Outputs:** Instead of embedding plots or large data in notebooks, **save them to the `outputs/` directory** (which is gitignored). You can then reference them in the notebook if needed, but the main file lives outside Git.

---

## 5. General Project Workflow (Streamlined)

1.  **Set Up:**
    - Create your virtual environment (`python -m venv .venv`).
    - Install dependencies (`pip install -r requirements.txt`).
    - Install and configure `nbdev` (`nbdev --install`).
    - Install and set up `pre-commit` hooks (`pip install pre-commit && pre-commit install`).
    - Download required data to the `data/` folder as described in `README.md`.
2.  **Develop in Notebooks:** Use notebooks for initial ideas and interactive development.
3.  **Refactor Code:** As soon as a piece of code becomes stable, reusable, or part of the core logic, **move it to `src/`**.
4.  **Create Scripts:** Develop standalone scripts in `scripts/` that orchestrate the main tasks (data prep, training, evaluation, full pipeline) by importing functions from `src/`. This allows for easy, reproducible runs without a notebook environment.
5.  **Run Tools:** Regularly run `black` and `isort` (or configure your IDE to do it automatically). `nbdev` will run on `git add`. The `pre-commit` hooks will run automatically on `git commit`.

---

## 6. Git & Version Control

- **No Git LFS:** Git LFS is strictly forbidden for this project. Large files must be managed externally and added to `.gitignore`.
- **Branching:** Work on feature branches (`feature/your-feature-name`, `bugfix/issue-description`) off of `main` and `dev`.
- **Commit Messages:** Write clear, concise, and descriptive commit messages.
- **Pull Requests (PRs):** Create PRs for all changes. Ensure your code adheres to these conventions, and that the PR description is clear.
- **Code Reviews:** All code should be reviewed before merging into `main` and `dev`.
