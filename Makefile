.PHONY: help test smoke clean

help:
	@echo "Available targets: test smoke clean"

test:
	@echo "Tests will be enabled after the environment milestone."

smoke:
	@echo "Smoke pipeline will be enabled after the environment milestone."

clean:
	rm -rf .pytest_cache htmlcov
