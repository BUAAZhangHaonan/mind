# MIND Progress Log

## 2026-03-28

- Bootstrapped the repository skeleton in `/home/d7049/zhanghaonan/mind`.
- Initialized local git on branch `feat/mind-bootstrap`.
- Deferred environment setup, data integration, and experiments to later milestones.
- Added `environment.yml`, `requirements.txt`, and `env.example` for the `mind-py311` runtime.
- Added `scripts/verify_env.py` for import, CUDA, and optional model-config checks.
- Added Make targets for environment creation and verification.
- Rebuilt the environment manifest after a partial failed solve truncated the file on disk.
- Updated the Hugging Face dependency pin to a version compatible with `transformers==4.57.1`.
- Observed that `conda env create` was unreliable on this machine and twice left broken partial environments.
- Switched the documented `make env` path to a verified two-step setup using `conda create` plus `pip install -r requirements.txt`.
- Verified the runtime with `conda run -n mind-py311 python scripts/verify_env.py`.
- Verified the first unit tests with `conda run -n mind-py311 pytest tests/unit/test_verify_env.py -q`.
