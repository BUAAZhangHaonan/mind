CONDA_BASE := $(shell conda info --base)
ENV_NAME := mind-py311
ENV_PYTHON := $(CONDA_BASE)/envs/$(ENV_NAME)/bin/python
CERT_BUNDLE := /etc/ssl/certs/ca-certificates.crt

.PHONY: help env verify test smoke clean

help:
	@echo "Available targets: env verify test smoke clean"

env:
	conda create -n $(ENV_NAME) -c conda-forge --override-channels -y python=3.11 pip git ca-certificates certifi openssl
	SSL_CERT_FILE=$(CERT_BUNDLE) REQUESTS_CA_BUNDLE=$(CERT_BUNDLE) PIP_CERT=$(CERT_BUNDLE) $(ENV_PYTHON) -m pip install -r requirements.txt

verify:
	conda run --no-capture-output -n $(ENV_NAME) python scripts/verify_env.py

test:
	conda run --no-capture-output -n $(ENV_NAME) pytest -q

smoke:
	@echo "Smoke pipeline will be enabled after the pipeline milestones."

clean:
	rm -rf .pytest_cache htmlcov
