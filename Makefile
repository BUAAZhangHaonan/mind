.PHONY: help env verify-env test smoke clean

help:
	@echo "Available targets: env verify-env test smoke clean"

env:
	conda create -n mind-py311 -c conda-forge --override-channels -y python=3.11 pip git ca-certificates certifi openssl
	SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt conda run -n mind-py311 python -m pip install -r requirements.txt

verify-env:
	conda run -n mind-py311 python scripts/verify_env.py

test:
	conda run -n mind-py311 pytest -q

smoke:
	@echo "Smoke pipeline will be enabled after the pipeline milestones."

clean:
	rm -rf .pytest_cache htmlcov
