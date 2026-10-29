# Optimization experiments

These measurements refer to the original selected epoch-3 model, SHA-256
`91228999587d6fdb9419c4b646b1dfd1a501faa22a2e8db848e15569293e1b8e`.
Later accuracy-training experiments must be reported separately; these numbers
must not be attributed to different weights without rerunning the measurements.

## Inference-only export

| Stored artifact | Bytes |
| --- | ---: |
| Full training checkpoint | 152,329,441 |
| Portable FP32 inference checkpoint | 76,406,459 |

Removing optimizer/training state reduces the stored file by 49.84%. This does
not quantize the model or reduce its in-memory parameter size. The exported
weights reproduce the source run's development score exactly (72.920848%).

## Region proposal count

Evaluation: all 1,144 development images, original size settings (320/640).
Latency: 80 fixed, evenly spaced development images, three repeats, five warmups
per setting, batch size one, paired interleaved execution order, four CPU threads.
Only model forward is timed; decoded tensors are prepared before timing.

| Proposals | Dev mAP@0.5 | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 150 | 72.92% | 68.12 | 62.71 | 96.86 |
| 75 | 72.31% | 68.92 | 60.37 | 94.37 |

The lower count loses 0.61 AP points and does not improve mean latency in this
run. The original 150 proposals remain the supported default. Small timing
differences are sensitive to background activity and thermal state; raw samples
and exact image IDs are in `optimization/latency.json`. These are single-machine
measurements, not service throughput guarantees.

## Dynamic INT8 linear heads

Method: `torch.ao.quantization.quantize_dynamic` on `nn.Linear` using QNNPACK.
Convolutions and all other operations remain FP32. Evaluation uses all development
images; latency uses 48 fixed dev images with three paired/interleaved repeats,
five warmups per model, and four CPU threads.

| Setting | State dict bytes | Dev mAP@0.5 | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 76,418,091 | 72.9208% | 60.68 | 60.51 | 64.38 |
| Dynamic INT8 linear heads | 34,417,691 | 72.9579% | 64.54 | 63.72 | 71.09 |

The comparable serialized state dictionaries shrink by 54.96%, but mean latency
increases by about 6.35%. The 0.037-point AP difference is tiny; it is not evidence
of a reliably better accuracy model. This is a useful storage/latency tradeoff,
not a successful speed optimization on this CPU. State-dict sizes here use the
same in-memory serialization method; small differences from exported-file sizes
come from metadata/serialization layout.

The pinned PyTorch still supplies this legacy API but emits deprecation and
QNNPACK warnings. The experiment is isolated in `voc_detection/quantization.py`;
the supported demo uses FP32 and does not depend on legacy quantized execution.
No ONNX, pruning, full-network INT8, or cross-platform performance claim is made.
Full scores and samples are in `optimization/int8/`.

## GPU training feasibility

The machine is an Apple M5 with 16 GB unified memory. MPS is available outside
the restricted execution sandbox. A temporary-model benchmark runs the same
training minibatch eight times per device and discards the first three timings:
CPU mean 0.3911 s/step, MPS mean 0.1615 s/step (about 2.42x faster). One MPS timed
step was 0.3887 s; the other four were 0.1015-0.1074 s. First-step GPU compilation
took about seven seconds. These are five fixed-batch measurements, not a full-epoch
speedup claim. Temporary benchmark updates were never used as trained checkpoints.

New GPU experiments continue from the actual epoch-3 checkpoint and are selected
on CPU development evaluation, matching the supported serving device. Horizontal
flips transform image pixels and both x coordinates together; they are never
applied to development or validation data. Tests verify both properties. GPU/CPU
randomness and numeric differences limit exact cross-device reproduction.
