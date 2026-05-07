ENV_NAME ?= mind-py311
CONDA ?= conda
CONDA_RUN ?= $(CONDA) run --no-capture-output -n $(ENV_NAME)
PYTHON ?= $(CONDA_RUN) python
PROJECT_PYTHONPATH := $(CURDIR)/src
CERT_BUNDLE ?= /etc/ssl/certs/ca-certificates.crt
HF_ENDPOINT ?= https://hf-mirror.com
MODEL_ID ?= Qwen/Qwen3-VL-8B-Instruct
SMOKE_MODELS ?= qwen3-vl-8b
SMOKE_DATASETS ?= pope
SMOKE_SUBSETS ?= popular
SMOKE_OUTPUT_ROOT ?= outputs/v2_stage0

.PHONY: help env install verify-env verify-model test plan-smoke clean

help:
	@echo "Available targets: env install verify-env verify-model test plan-smoke clean"

env:
	$(CONDA) create -n $(ENV_NAME) -c conda-forge --override-channels -y python=3.11 pip git ca-certificates certifi openssl
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -r requirements.txt
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -e .

install:
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -e .

verify-env:
	PYTHONPATH=$(PROJECT_PYTHONPATH)$${PYTHONPATH:+:$${PYTHONPATH}} $(PYTHON) scripts/verify_env.py

verify-model:
	HF_ENDPOINT=$(HF_ENDPOINT) PYTHONPATH=$(PROJECT_PYTHONPATH)$${PYTHONPATH:+:$${PYTHONPATH}} $(PYTHON) scripts/verify_env.py --model-id $(MODEL_ID)

test:
	PYTHONWARNINGS=ignore $(PYTHON) -m pytest -q tests/unit tests/v2

plan-smoke:
	$(PYTHON) scripts/v2/stage0_run.py --models $(SMOKE_MODELS) --datasets $(SMOKE_DATASETS) --subsets $(SMOKE_SUBSETS) --output-root $(SMOKE_OUTPUT_ROOT) --dry-run

clean:
	rm -rf .pytest_cache htmlcov src/*.egg-info
