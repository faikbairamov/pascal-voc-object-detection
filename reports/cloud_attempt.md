# Free Colab GPU attempt

A free Google Colab Tesla T4 runtime was successfully allocated and CUDA was
verified. Runtime PyTorch was 2.11.0+cu130. With explicit user approval, an 825 MB
bundle containing source, a full training checkpoint, and public VOC fit/dev
images was uploaded. Validation images were not included.

The experiment attempted fixed-384 training, batch size 4, horizontal flips,
SGD LR 0.0001 and CUDA mixed precision from the epoch-4 flip checkpoint. The
initial epoch progressed to at least batch 900 of 1,144 but slowed substantially,
with system RAM approaching the free runtime's 12.67 GB limit. A utilization
snapshot showed 0% GPU and a process snapshot about one busy CPU core; these
snapshots alone do not establish the root cause. The runtime did not produce
a completed, scored checkpoint before it was stopped to protect the deadline.
No cloud accuracy result or cloud speedup is claimed.

The successful local Apple M5/MPS experiments remained available throughout.
Their completed checkpoints, MLflow measurements and full development evaluations
were used for final model selection. More expensive hardware being offered for
free does not establish a faster end-to-end run for this workload and environment.

Cloud launch source is retained under `tools/cloud_train.py` for future review;
its cuDNN benchmarking and free-runtime memory behavior require investigation
before treating it as a reliable recipe. No paid compute or subscription was used.
