# Selected inference model

`best.pt` is supplied in the [portable submission bundle](https://github.com/faikbairamov/pascal-voc-object-detection/releases/latest/download/pascal-voc-submission.zip). It is ignored by Git.
`best.metadata.json` identifies the class order, source training run, epoch,
training configuration, inference settings, file size and SHA-256 checksum.

Use the supplied file with `python -m voc_detection.demo` or
`python -m voc_detection.inference /path/to/photo.jpg`. Loading weights does not
require the dataset or a pretrained-weight download. Only load trusted checkpoint
files. The inference loader uses `torch.load(..., weights_only=True)`.

If using a Git checkout without the accompanying binary, follow the training and
export instructions in the main README. Training checkpoints containing optimizer
state are distinct from this inference-only export.
