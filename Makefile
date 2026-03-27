ENV_NAME ?= mind-py311
ENV_PREFIX ?= /tmp/$(ENV_NAME)
PYTHON ?= $(ENV_PREFIX)/bin/python
CONDA ?= conda
CERT_BUNDLE ?= /etc/ssl/certs/ca-certificates.crt
HF_ENDPOINT ?= https://hf-mirror.com
MODEL_ID ?= Qwen/Qwen3-VL-8B-Instruct
SMOKE_CONFIG ?= configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml

.PHONY: help env install verify-env verify-model test plan-smoke clean

help:
	@echo "Available targets: env install verify-env verify-model test plan-smoke clean"

env:
	$(CONDA) create -p $(ENV_PREFIX) -c conda-forge --override-channels -y python=3.11 pip git ca-certificates certifi openssl
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -r requirements.txt
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -e .

install:
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(PYTHON) -m pip install -e .

verify-env:
	$(PYTHON) scripts/verify_env.py

verify-model:
	HF_ENDPOINT=$(HF_ENDPOINT) $(PYTHON) scripts/verify_env.py --model-id $(MODEL_ID)

test:
	PYTHONWARNINGS=ignore $(PYTHON) -m pytest -q tests/unit tests/integration

plan-smoke:
	$(PYTHON) scripts/run_experiment.py --config $(SMOKE_CONFIG) --stages all

clean:
	rm -rf .pytest_cache htmlcov src/*.egg-info
