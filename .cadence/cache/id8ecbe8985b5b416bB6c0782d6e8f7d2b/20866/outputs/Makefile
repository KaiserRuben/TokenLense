.PHONY: install test lint format clean run-tests run-benchmark

# Installation
install:
	pip install -r requirements.txt

# Testing
test:
	python -m pytest

run-tests:
	python tests/inseq/test_inseq_methods.py

# Linting and formatting
lint:
	black --check src tests work
	mypy src tests work

format:
	black src tests work

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

# Run in Cadence
cadence-tests:
	cadence run .cadence/configs/inseq_tests.yaml

cadence-benchmark:
	cadence run .cadence/configs/attribution_benchmark.yaml

# Create directories if they don't exist
init:
	mkdir -p output/data
	mkdir -p output/graphs
	mkdir -p output/benchmark