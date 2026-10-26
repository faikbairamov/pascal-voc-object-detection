# Initial baseline: development results

Architecture: TorchVision Faster R-CNN MobileNetV3-Large 320 FPN, COCO-pretrained,
with a new 21-class predictor. CPU, batch size 2, SGD learning rate 0.001,
momentum 0.9, weight decay 0.0005, seed 42. Difficult objects are included during training.

| Completed epoch | Training loss | Training minutes | Development mAP@0.5 |
| --- | --- | --- | --- |
| 1 | 1.0607 | 16.1 | 69.84% |
| 2 | 0.8142 | 15.8 | 70.66% |

These rounded results were reported in the original notebook/user session, not rerun
as part of workspace reorganization. Run: `d43c960633a7477e8e8d719de3a27338`.
Both checkpoints remain under the ignored `mlflow_local/checkpoints/` directory.

## Evaluation protocol

- Fit: 4,573 images from the official train split; development: 1,144 images.
- Final local evaluation: official val (5,823 images), still held out.
- Custom VOC all-points interpolated AP with strict IoU > 0.5 matching.
- Non-difficult annotations form the recall denominator; matches to difficult objects are ignored.
- Scores are computed on the detector's returned predictions, including its default score filtering,
  NMS, and per-image detection cap. No additional 0.20 display filter is used for AP.
- COCO pretraining supplies prior object knowledge; these are transfer-learning results.
- Metric tests cover basic matching cases, not official-devkit numerical parity on a full dataset.

## Observed errors

Example bottle images show missed small bottles, can/bottle confusion, duplicate boxes,
and background false positives. Three examples illustrate failure modes, not their frequency.
Bottle AP moved from 34.30% to 33.75%; pottedplant AP from 30.11% to 34.73%.

## Reproducibility limits

Original checkpoints retain model/optimizer state but not random-generator state;
continuation after a kernel restart cannot reproduce the original shuffled order exactly.
New training notebooks save PyTorch and loader RNG state, but do not yet implement a resume workflow.
The local MLflow database/artifacts are not in Git; share exported reports and intentionally
published weights separately. No final-test claim is made from development results.

AI assistance was used to explain and structure code and workflow. The final submission
should describe its use and the author's own implementation and verification accurately.
