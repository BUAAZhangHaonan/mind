ENV_PREFIX := /home/d7049/.conda/envs/mind-py311
ENV_PYTHON := $(ENV_PREFIX)/bin/python

.PHONY: help env verify-env test smoke clean

help:
	@echo "Available targets: env verify-env test smoke clean"

env:
	conda env create -p $(ENV_PREFIX) -f environment.yml
	$(ENV_PYTHON) -m pip install -r requirements.txt

verify-env:
	$(ENV_PYTHON) scripts/verify_env.py

test:
	$(ENV_PYTHON) -m pytest -q

smoke:
	@echo "Smoke pipeline will be enabled after the pipeline milestones."

clean:
	rm -rf .pytest_cache htmlcov
