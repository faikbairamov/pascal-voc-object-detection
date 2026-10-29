# Submission guide

The deliverable is a source repository plus a portable bundle containing the
selected trained weights. Use the final result files and model metadata as the
source of truth; experimental and baseline scores are labelled separately.

| Requirement from the assignment | Delivered evidence |
| --- | --- |
| Source code, dependencies, training/use instructions | `README.md`, `voc_detection/`, requirements, config, splits, notebooks and tests |
| Trained model and new-image demonstration | `models/best.pt` in the bundle, model metadata, CLI and FastAPI browser demo |
| Experiment comparison and task metrics | `reports/experiments.csv`, `reports/technical_report.md`, evaluation category CSVs |
| Optimization before/after | `reports/optimization.md`, raw proposal/INT8/device results |
| Short technical report | `reports/technical_report.md`; shareable PDF under `output/pdf/` when generated |
| AI tool disclosure | `reports/ai_use.md` |

## Reviewer quick start

1. Extract the supplied ZIP. It includes the model; a Git checkout by itself
   excludes binary weights intentionally.
2. Follow the Python 3.14 setup in README, or build the Docker image.
3. Start `python -m voc_detection.demo` in the configured environment, open
   `http://127.0.0.1:7860`, and upload a new image. `/docs` exposes the API.
4. Read the technical report and the experiment/optimization evidence.
5. To reproduce evaluation or train again, obtain VOC 2012 train/val and follow
   the data placement and CLI instructions. Inference needs no dataset.

The ZIP includes a SHA-256 manifest for source, reports, and weights. Its external
`.sha256` file verifies archive transfer integrity. Never upload the virtual
environment, raw dataset, local MLflow database, or full optimizer checkpoints.

## Packaging command

Run from the repository root after completing final verification:

```bash
.venv/bin/python tools/package_submission.py
```

This creates `dist/pascal-voc-submission.zip` and an accompanying checksum. The ZIP
is intentionally ignored by Git. The repository is [pascal-voc-object-detection](https://github.com/faikbairamov/pascal-voc-object-detection).
The [latest release](https://github.com/faikbairamov/pascal-voc-object-detection/releases/latest)
provides the complete ZIP with trained weights and checksum. Source-only Git
checkouts intentionally omit the binary checkpoint. Packaging alone does not
publish; the release must be uploaded before sending the repository link.

## Defense preparation

Read `reports/defense_notes.md` and review the actual code. Be ready to explain
coordinate conversion, difficult objects, AP matching, transfer learning, learning
rate and augmentation experiments, GPU/CPU differences, and how to start the API.
Do not describe AI-assisted work as independently written or reviewed unless that
is accurate.
