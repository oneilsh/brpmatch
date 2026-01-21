.PHONY: help install dev clean build check test test-quick example zip publish

default: help

# Java 17 is required for local Spark (Java 23+ has Arrow compatibility issues)
# macOS (Homebrew): brew install openjdk@17
# Linux: apt install openjdk-17-jdk or equivalent
JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX := /usr/lib/jvm/java-17-openjdk-amd64

# Auto-detect: use JAVA_HOME if set, otherwise try macOS path, then Linux path
JAVA_HOME ?= $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; elif [ -d "$(JAVA_HOME_LINUX)" ]; then echo "$(JAVA_HOME_LINUX)"; fi)

help:
	@echo "Common development commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make dev          - Install dev dependencies (including build tools)"
	@echo "  make test         - Run pytest with local Spark"
	@echo "  make test-quick   - Run pytest in fast-fail mode"
	@echo "  make example      - Run example pipeline with lalonde dataset"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make build        - Build sdist and wheel"
	@echo "  make zip          - Create importable .zip for offline use"
	@echo "  make check        - Check build with twine"
	@echo "  make publish-test - Upload to TestPyPI"
	@echo "  make publish      - Upload to PyPI (live, not test)"
	@echo ""
	@echo "Note: Java 17 required for local Spark. Install with:"
	@echo "  macOS: brew install openjdk@17"
	@echo "  Linux: apt install openjdk-17-jdk"

install:
	poetry install

dev:
	poetry install
	poetry run pip install build twine

clean:
	rm -rf dist/ build/ *.egg-info

test:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v

test-quick:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -x --tb=short

example:
	JAVA_HOME=$(JAVA_HOME) poetry run python example/example.py

zip: clean build
	mkdir -p dist
	zip -r dist/brpmatch.zip brpmatch -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Created dist/brpmatch.zip"

build:
	poetry run python -m build

check:
	poetry run twine check dist/*.whl dist/*.tar.gz

publish-test: build check
	@echo "Publishing to TestPyPI..."
	# Export env vars so twine picks them up
	. .env && \
	TWINE_USERNAME=__token__ TWINE_PASSWORD=$$TEST_PYPI_TOKEN poetry run twine upload --repository testpypi dist/*.whl dist/*.tar.gz --skip-existing

publish: build check
	@echo "Publishing to PyPI..."
	. .env && \
	TWINE_USERNAME=__token__ TWINE_PASSWORD=$$PYPI_TOKEN poetry run twine upload dist/*.whl dist/*.tar.gz --skip-existing
