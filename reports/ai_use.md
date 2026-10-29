# AI tools disclosure

ChatGPT/Codex assisted with explanations of the object-detection workflow,
organization/refactoring of Python modules and notebooks, metric reasoning,
and preparation of the submission. During the final packaging session, Codex
implemented the shared inference path, FastAPI upload demo, Docker serving setup,
CLI training/evaluation/export tools, automated checks, analysis figures, and
documentation with the author's direction.

Verification actually performed includes a full image/XML/split audit, fresh development and held-out validation evaluations of the selected exported
weights (scores and hashes are in `reports/evaluation/`),
synthetic metric and API contract checks, real-model CLI inference, browser upload
verification, a built Linux ARM64 Docker image, and real-image API smoke tests.
Optimization measurements and per-class/structured errors are saved as result files.
The original CPU training and metrics came from the existing project. With the
author's explicit request to improve accuracy, Codex ran additional tracked MPS
fine-tuning and resolution/proposal experiments. A user-approved free Colab T4
attempt did not finish a scored checkpoint and was stopped for the deadline.
Only completed measured runs are reported as training results; no metrics are
invented. See the technical report for final selection and limitations.
