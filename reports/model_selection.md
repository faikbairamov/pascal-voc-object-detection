# Final model selection before final validation

Decision recorded: 2026-10-29T14:32:44+01:00.

Selected training run: `63411f882d0d45898a5877fa40e2c0c9`, epoch 7.
Selected development score: 74.94341279%, all 1,144 fixed dev images.
Inference settings: minimum side 384, maximum side 768, 300 pre/post-NMS RPN
proposals, default per-image detection cap. Display threshold 0.5; no additional
display threshold is applied to AP.

Final inference checkpoint SHA-256: `f4c5318a5e574007e9473894f277333a302252d7e0f28514499543cbce70cb5c`.
Source training checkpoint SHA-256: `43db303aadb6be96aa0159202af7378893b8cb2eb024f6011ad4214c29ed16f8`.

The original best epoch-3 model scored 72.920848% on dev. Extra MPS training,
horizontal flips, mixed/fixed scales and dev inference sweeps led to the selected
74.943413% candidate. `reports/selection.json` records the ranked completed dev
sweeps; `reports/experiments.csv` records training results at their logged eval
settings. Scores at 320 and 384 pixels must not be conflated.

All model selection for this final checkpoint used development. The earlier
selected epoch-3 model was already evaluated on official val (71.842046%) before
the user requested additional accuracy work. The final selected model is now
frozen for its own validation evaluation; no further training or selection uses
that final score. Thus val remains held out from fitting, but should not be
portrayed as an untouched independent test after all iterations. No official VOC
test-server score is claimed.

The free Colab T4 attempt was stopped without a completed scored checkpoint and
contributed no selected weights. See `reports/cloud_attempt.md`.
