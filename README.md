# Pascal VOC 2012 object detection

Fine-tune a COCO-pretrained Faster R-CNN MobileNetV3-Large 320 FPN detector on
all 20 VOC object categories. The current work includes training, development
AP evaluation, local MLflow tracking, and visual error analysis.

## Project layout

```text
configs/baseline.json        Shared paths and baseline hyperparameters
voc_detection/
  dataset.py                VOC XML parsing, image tensors, and batching
  model.py                  Pretrained detector and VOC predictor construction
  engine.py                 One training epoch and fresh prediction collection
  metrics.py                VOC AP and difficult/duplicate matching rules
  evaluation.py             All-category evaluation and summary tables
  visualization.py          Category-level error inspection
  project.py                Repository paths, seeds, and loader construction
  tracking.py               Local MLflow setup and evaluation logging
notebooks/
  01_data_exploration.ipynb  Dataset inspection and saved split generation
  02_sanity_checks.ipynb     Dataset checks and tiny-batch learning checks
  03_training.ipynb          Explicit new training experiments with epoch checkpoints
  04_evaluation.ipynb        Reload saved weights, score development, inspect errors
  archive/                  Original workbook with every saved output preserved
splits/                     Version-controlled fitting and development image IDs
reports/baseline.md          Recorded epoch results, error observations, and limitations
tests/test_metrics.py       Small synthetic evaluator regression checks
mlflow_local/               Ignored local database, artifacts, and checkpoints
```

## Environment

The existing `.venv` is retained. Its observed Python version is 3.14.8;
`requirements.txt` pins the direct dependencies actually used locally. These
versions have not been validated on other platforms. Select the Pascal VOC
`.venv` kernel in VS Code. Notebooks locate the repository root automatically
when opened from the root or `notebooks/`; they import reusable code rather
than redefining the training and metric implementations.

For a new environment, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Optional package installation: `.venv/bin/python -m pip install -e .`.
Installation was not performed during reorganization. Restart the notebook kernel
after editing imported Python modules so stale definitions do not remain in memory.

## Data and splits

Place the dataset beside this repository:

```text
workspace/
  VOCdevkit/VOC2012/{JPEGImages,Annotations,ImageSets,...}
  pascal-voc-object-detection/
```

Change `dataset_dir` in `configs/baseline.json` for another location (paths are
resolved relative to this repository). Dataset files and TAR archives stay out
of Git. Keep the existing `splits/train.txt` and `splits/dev.txt` assignments:
4,573 fitting images and 1,144 development images. The official validation split
of 5,823 images is reserved for final local evaluation. Official test labels are
not part of this project.

## What to open next

The completed baseline already has two epochs; do not retrain just to migrate the
workspace. Start with **`notebooks/04_evaluation.ipynb`** to load its epoch-2
checkpoint. This notebook does not train or log to MLflow automatically.
Notebook 03 starts a new model and a new experiment and contains a clearly marked
training cell. Run initialization again before a fresh experiment, rather than
rerunning only its training cell on already-updated weights.

Notebook 01 can write split manifests; do not change their seed or assignments
mid-experiment. Notebook 02 trains only a small temporary sanity-check model.
The archived workbook is historical reference, not an executable entry point.

## Tracking

Keep the current `mlflow_local/` directory where it is: existing experiment
artifact locations refer to that directory. From the repository root:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow_local/tracking.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Existing run and checkpoint files are preserved.
The recorded development score rose from 69.84% to 70.66% mAP@0.5 after two epochs;
see `reports/baseline.md` for protocol and limitations.

## Checks

Run the small synthetic metric tests without training or downloading weights:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Code, notebooks, configs, splits, tests, and reports belong in Git. Dataset images,
virtual environments, checkpoints, and local MLflow databases do not. No Git commit
or push is performed automatically.
