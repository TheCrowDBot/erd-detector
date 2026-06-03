# ERD YOLO

Object detection pipeline for Entity Relationship Diagram (ERD) symbol detection using YOLO.

## Overview

This repository contains the training, validation, evaluation, and benchmarking pipeline used to detect ERD components and relationship cardinalities.

Supported classes:

* entity
* attribute
* rel_one
* rel_many
* rel_zero_or_one
* rel_zero_or_many
* rel_one_or_many
* rel_one_only
* inheritance

The project is organized around reproducible experiments, dataset versioning, and staged training workflows.

## Repository Structure

```text
.
├── configs/
│   ├── datasets/
│   ├── experiments/
│   └── train/
│
├── datasets/
│
├── runs/
│
├── src/
│   ├── pipeline/
│   │   ├── dataset.py
│   │   ├── stage1.py
│   │   ├── stage2.py
│   │   ├── validate.py
│   │   ├── evaluate.py
│   │   ├── benchmark.py
│   │   ├── report.py
│   │   └── full_train.py
│   │
│   ├── utils/
│   └── cli.py
│
├── experiments/
├── notebooks/
└── README.md
```

## Pipeline

The training workflow is divided into independent stages.

### Dataset

Downloads and prepares data from the annotation platform.

Responsibilities:

* Download project exports
* Extract archives
* Split train/validation datasets
* Generate YOLO-compatible dataset structure
* Cleanup temporary files

```bash
python src/cli.py dataset
```

---

### Stage 1 Training

Initial training using pretrained YOLO weights.

Responsibilities:

* Dataset preparation
* Training configuration loading
* Model training
* Run registration

```bash
python src/cli.py stage1
```

Output:

```text
runs/stage1/
```

---

### Stage 2 Training

Transfer learning using the best checkpoint produced during Stage 1.

Responsibilities:

* Dataset preparation
* Weight loading
* Fine-tuning
* Run registration

```bash
python src/cli.py stage2
```

Output:

```text
runs/stage2/
```

---

### Validation

Runs validation against a trained model checkpoint.

Responsibilities:

* Validation metrics
* Precision and recall calculation
* mAP computation

```bash
python src/cli.py validate
```

---

### Evaluation

Generates detailed per-class performance metrics.

Responsibilities:

* Class-level precision
* Class-level recall
* Confusion matrix generation
* Detection analysis

```bash
python src/cli.py evaluate
```

---

### Benchmarking

Compares multiple experiment runs.

Responsibilities:

* Cross-run metric comparison
* Model ranking
* Regression detection

```bash
python src/cli.py benchmark
```

---

### Reporting

Generates experiment summaries.

Responsibilities:

* Training configuration summary
* Dataset information
* Validation results
* Evaluation metrics

```bash
python src/cli.py report
```

---

### Full Training Pipeline

Executes the complete workflow.

```bash
python src/cli.py full-train
```

Execution order:

```text
Dataset Preparation
        ↓
Stage 1 Training
        ↓
Stage 2 Training
        ↓
Validation
        ↓
Evaluation
        ↓
Report Generation
```

## Experiments

All experiments should be defined through configuration files.

Example:

```yaml
name: stage2_transfer

dataset:
  project_id: 13
  train_ratio: 0.8
  val_ratio: 0.2
  seed: 4332234443

training:
  config: configs/train/stage2.yaml

model:
  base_weights: runs/stage1/weights/best.pt
```

Running an experiment:

```bash
python src/cli.py stage2 --config experiments/stage2_transfer.yaml
```

## Run Tracking

Each training run should produce a self-contained directory containing:

```text
runs/
└── train-001/
    ├── config.yaml
    ├── metrics.json
    ├── report.md
    ├── weights/
    │   └── best.pt
    └── artifacts/
```

This ensures experiments remain reproducible and comparable over time.

## Development

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## License

See LICENSE for licensing information.
