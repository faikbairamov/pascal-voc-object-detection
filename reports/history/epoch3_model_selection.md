# Model selection decision before final validation

Selected weights: run `1e1ca85634974b3289a6dac12ea3df12`, epoch 3, learning rate
0.0001 for continuation from baseline epoch 2. Freshly recomputed development
mAP@0.5: 0.7292084813882795 on all 1,144 development images.

Freeze the deployment configuration at a 320-pixel minimum side, 640-pixel maximum
side, 150 pre/post-NMS RPN proposals, and the default 100 detections per image.
The display confidence defaults to 0.5; evaluation uses all returned detections.

The 75-proposal experiment scored 0.7230930479615159 on development. Its paired
CPU latency benchmark did not improve mean latency, so it is rejected for the
main demo. Inference-only export removes optimizer state, halving the stored
checkpoint size without changing predictions or reducing numerical precision.

The official validation labels are used next for final local evaluation only;
no training or model selection will use that score. This is a local held-out
benchmark, not an official challenge test-server result. Pretraining provenance
is COCO; overlap with pretraining data has not been independently audited.
