# Historical baseline and experiment provenance

The first two epochs used Faster R-CNN MobileNetV3-Large 320 FPN, COCO pretrained,
with a 21-class predictor, CPU, batch size 2, SGD LR 0.001, momentum 0.9,
weight decay 0.0005, seed 42, and difficult annotations included during training.

| Epoch | Training loss | Training minutes | Development mAP@0.5 |
| --- | --- | --- | --- |
| 1 | 1.0607 | 16.1 | 69.84% |
| 2 | 0.8142 | 15.8 | 70.66% |
| 3, control continuation | 0.7139 | 15.6 | 69.99% |
| 3, LR 0.0001 continuation | 0.6203 | 15.5 | **72.92%** |

Run `d43c960633a7477e8e8d719de3a27338` contains the baseline/control;
run `1e1ca85634974b3289a6dac12ea3df12` contains the reduced-LR continuation.
Both continuations restart seed-42 shuffle from the same epoch-2 checkpoint.
The original first two checkpoints do not preserve RNG state. New checkpoints
preserve Torch and loader RNG state; the CLI supports normal resume or an explicit
shuffle restart. Original training/checkpoint files and notebook outputs are retained.

The source for this table is local MLflow metric history, exported to
`reports/experiments.csv`. The selected 72.92% development score was independently
recomputed during submission preparation. This epoch-3 model's held-out
validation result was **71.84%**; see `reports/technical_report.md` for the protocol,
optimization comparisons, errors and limitations. These are local custom-VOC scores,
not official test-server scores.
