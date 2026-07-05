# ERD Detector

ERD Detector is a Python project for training and evaluating object detection models that identify Entity Relationship Diagram (ERD) symbols and relationship cardinalities with YOLO-based pipelines.

## What this project does

The repository contains a reproducible workflow for:

- downloading labeled datasets from Label Studio,
- splitting data into train/validation sets,
- training a base model,
- applying transfer learning from a previous checkpoint,
- evaluating the resulting model and collecting metrics.

The current pipeline is designed around YOLO object detection and supports classes such as:

- entity
- attribute
- rel_one
- rel_many
- rel_zero_or_one
- rel_zero_or_many
- rel_one_or_many
- rel_one_only
- inheritance

## Repository layout

```text
.
├── configs/
│   ├── train/
│   └── val/
├── experiments/
├── src/
│   ├── config/
│   ├── experiments/
│   ├── pipeline/
│   ├── schemas/
│   └── utils/
└── pyproject.toml
```

## Prerequisites

- Python 3.10+
- A working Label Studio instance
- Environment variables for Label Studio access:

```bash
export LABEL_STUDIO_URL="https://your-label-studio-instance"
export LABEL_STUDIO_API_KEY="your-api-key"
```

You can also place these values in a `.env` file if your environment loads it automatically.

## Quick start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the project dependencies required by your environment.

3. Run the end-to-end example workflow:

```bash
poe model
```

This entry point performs a full example flow:

1. initial training,
2. transfer learning from the trained checkpoint,
3. evaluation on a validation/evaluation dataset.

## Available workflows

### Training

Run the standard training pipeline:

```bash
poe train
```

### Transfer learning

Run transfer learning using a pretrained checkpoint:

```bash
poe transfer
```

### Evaluation

Evaluate a trained model:

```bash
poe evaluate
```

### Sweeps and experiments

The repository also includes experiment-oriented runners for more advanced workflows:

```bash
poe experiment
```

## Configuration

Training and validation settings are stored under the configuration directories:

- [configs/train](configs/train)
- [configs/val](configs/val)

The pipeline reads model, dataset, and downloader settings from the modules in [src/config](src/config).

## Output

Training and evaluation runs generate artifacts under the project workspace, including model checkpoints and run-specific metrics. The exact output location depends on the training configuration and the run context.

## Development notes

The project includes a small task setup in [pyproject.toml](pyproject.toml) for common maintenance commands such as formatting, linting, and type checking.

## License

A license file is not currently included in the repository, so usage terms should be defined before sharing or redistributing the project.
