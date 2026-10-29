# Technical defense preparation

Use these as review prompts. Be able to explain the implementation in your own
words and locate the relevant functions; these notes do not establish that you
have already reviewed or independently written the code.

1. **Trace one image end to end.** XML coordinates become an RGB tensor and a
   target dictionary. TorchVision resizes/normalizes, MobileNet/FPN extracts
   features, RPN proposes boxes, ROI heads classify/refine, and postprocessing
   returns original-image coordinates. Inference adds confidence filtering and
   draws boxes without changing the AP evaluation protocol.
2. **Explain the losses.** Faster R-CNN supplies classification and box-regression
   losses for the ROI heads, plus objectness and box-regression losses for the
   RPN. The training loop sums them, clears gradients, backpropagates, and takes
   an SGD step. Momentum and weight decay have distinct roles.
3. **Explain the split.** Whole images, seed 42, 4,573 fit / 1,144 development.
   Development selects experiments; official val measures the selected model.
   COCO pretraining is disclosed. Image-ID isolation is not a proof against
   pretraining overlap or near-duplicate photographs.
4. **Explain AP, not just its value.** Per-category confidence ranking, strict
   IoU > 0.5, one match per non-difficult object, duplicates penalized, difficult
   matches ignored, interpolated precision envelope integrated over recall,
   and equal averaging across 20 classes. A 0.5 demo score threshold is not an
   IoU threshold, and neither is the mAP percentage.
5. **Defend an experiment.** Compare original epoch 2, the reset-shuffle epoch-3
   control, and the lower-LR continuation. Distinguish a historical tracked result
   from a freshly recomputed one. Understand why lower training loss alone is
   insufficient, and why one seed does not prove significance.
6. **Explain the GPU experiments.** Apple M5 uses MPS, not CUDA. A warmup matters;
   a fixed-batch microbenchmark does not equal full-epoch speed. All candidate
   scores use CPU inference on the same development images for deployment parity.
7. **Show an error.** Inspect bottle/pottedplant examples. At confidence 0.5,
   small-object recall is particularly weak. Describe the original-image area
   convention and why this diagnostic is different from COCO AP-small.
8. **Defend optimization choices.** Removing optimizer state reduces the file,
   not compute. Fewer proposals reduced AP with no mean speed gain in the initial
   benchmark. Partial INT8 reduced storage but slowed this CPU. A failed speed
   optimization is still informative when measured and disclosed.
9. **Demonstrate the API.** Start FastAPI, open `/docs`, upload a new image, change
   the threshold, and show errors for invalid bytes. Explain why model loading
   happens once and why access is serialized. Show the Docker build/run commands.
10. **Practice a code change.** Add a display threshold, another response field,
    or an image-size validation rule; identify tests that should change. Locate
    class mapping, box transforms, training flags, config and checkpoint loading.

Before submitting, read the AI disclosure and describe your contributions honestly.
Review `reports/technical_report.md` and the actual final experiment result files;
use the final selected scores rather than memorizing earlier baseline numbers.
