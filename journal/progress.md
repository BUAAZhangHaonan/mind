# MIND Progress Log

## 2026-03-28

- Bootstrapped the repository skeleton in `/home/d7049/zhanghaonan/mind`.
- Initialized local git on branch `feat/mind-bootstrap`.
- Created the project environment at `/home/d7049/miniconda3/envs/mind-py311`.
- Installed core dependencies including PyTorch 2.6.0 with CUDA 12.4 wheels, Transformers 4.57.1, Datasets 3.5.0, scikit-learn 1.6.1, and PyWavelets 1.8.0.
- Verified imports and GPU visibility with `scripts/verify_env.py`.
- Notes:
  - `conda run` was unreliable on this machine, so the repository now uses the environment's Python interpreter directly in `Makefile`.
  - The base `defaults` channel required a local ToS acceptance even though the project environment file uses `conda-forge`.
- Added `environment.yml`, `requirements.txt`, and `env.example` for the `mind-py311` runtime.
- Added `scripts/verify_env.py` for import, CUDA, and optional model-config checks.
- Added Make targets for environment creation and verification.
- Rebuilt the environment manifest after a partial failed solve truncated the file on disk.
- Updated the Hugging Face dependency pin to a version compatible with `transformers==4.57.1`.
- Observed that `conda env create` was unreliable on this machine and twice left broken partial environments.
- Switched the documented `make env` path to a verified two-step setup using `conda create` plus `pip install -r requirements.txt`.
- Verified the runtime with `/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/verify_env.py`.
- Verified the first unit tests with `/home/d7049/miniconda3/envs/mind-py311/bin/python -m pytest tests/unit/test_verify_env.py -q`.
