# Makefile for PFM MIP-based Local Search Project
# Cross-platform support for Windows, Linux, and macOS

# Project configuration
PROJECT_NAME := pfm-mip-based-local-search
VENV_NAME := venv
PYTHON := python
PIP := pip

# Platform detection
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    VENV_BIN := $(VENV_NAME)/Scripts
    PYTHON_VENV := $(VENV_BIN)/python.exe
    PIP_VENV := $(VENV_BIN)/pip.exe
    ACTIVATE := $(VENV_BIN)/activate.bat
    RM := rmdir /s /q
else
    DETECTED_OS := $(shell uname -s)
    VENV_BIN := $(VENV_NAME)/bin
    PYTHON_VENV := $(VENV_BIN)/python
    PIP_VENV := $(VENV_BIN)/pip
    ACTIVATE := $(VENV_BIN)/activate
    RM := rm -rf
endif

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available make targets (summary)
	@echo "PFM MIP-based Local Search - Make targets"
	@echo "========================================="
	@echo "Environment:   venv | clean-venv"
	@echo "Install:       install | install-dev | requirements"
	@echo "Quality:       fmt"
	@echo "Run:           run | run-vallada"
	@echo "Build:         build | clean-build"
	@echo "Cleanup:       clean | clean-all"
	@echo "Info:          info"

##@ Information
.PHONY: info
info:  ## Show project information
	@echo "Project Information"
	@echo "=================="
	@echo "Project Name: $(PROJECT_NAME)"
	@echo "Python: $(PYTHON)"
	@echo "Platform: $(DETECTED_OS)"
	@echo "Virtual Environment: $(VENV_NAME)"
	@echo "Virtual Environment Exists: $(if $(wildcard $(VENV_BIN)),Yes,No)"
ifeq ($(wildcard $(VENV_BIN)),$(VENV_BIN))
	@echo "Python Version in venv:"
	@$(PYTHON_VENV) --version
	@echo "Installed packages:"
	@$(PIP_VENV) list
endif

##@ Environment Setup
.PHONY: venv
venv:  ## Create virtual environment
	@echo "Creating virtual environment for $(DETECTED_OS)..."
ifeq ($(DETECTED_OS),Windows)
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created at $(VENV_NAME)"
	@echo "To activate: $(ACTIVATE)"
else
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created at $(VENV_NAME)"
	@echo "To activate: source $(ACTIVATE)"
endif

.PHONY: clean-venv
clean-venv:  ## Remove virtual environment
	@echo "Removing virtual environment..."
	$(RM) $(VENV_NAME)

##@ Installation
.PHONY: install
install:  ## Install package in editable mode (pip install -e .)
	@echo "Installing package in editable mode..."
ifeq ($(wildcard $(VENV_BIN)),)
	@echo "Virtual environment not found. Creating one..."
	$(MAKE) venv
endif
	$(PIP_VENV) install -e .

.PHONY: install-dev
install-dev:  ## Install package with development dependencies
	@echo "Installing package with development dependencies..."
ifeq ($(wildcard $(VENV_BIN)),)
	@echo "Virtual environment not found. Creating one..."
	$(MAKE) venv
endif
	$(PIP_VENV) install -e ".[dev]"

.PHONY: requirements
requirements:  ## Generate requirements.txt from pyproject.toml
	@echo "Generating requirements.txt..."
	$(PIP_VENV) install pip-tools
	$(PIP_VENV) freeze > requirements.txt

##@ Code Quality
.PHONY: fmt
fmt:  ## Format code with black and isort
	@echo "Formatting code..."
ifeq ($(wildcard $(VENV_BIN)),)
	@echo "Virtual environment not found. Installing dev dependencies..."
	$(MAKE) install-dev
endif
	$(PYTHON_VENV) -m black src/ --line-length 90
	$(PYTHON_VENV) -m isort src/ --profile black

##@ Development Workflow
.PHONY: dev-setup
dev-setup: venv install-dev  ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "Virtual environment: $(VENV_NAME)"
ifeq ($(DETECTED_OS),Windows)
	@echo "Activate with: $(ACTIVATE)"
else
	@echo "Activate with: source $(ACTIVATE)"
endif

##@ Cleanup
.PHONY: clean
clean: clean-build  ## Clean all generated files
	@echo "Cleaning all generated files..."
	find . -type d -name "*.egg-info" -exec $(RM) {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-venv  ## Clean everything including virtual environment
	@echo "Deep clean completed!"

##@ Build and Distribution
.PHONY: build
build:  ## Build distribution packages
	@echo "Building distribution packages..."
ifeq ($(wildcard $(VENV_BIN)),)
	@echo "Virtual environment not found. Installing build dependencies..."
	$(MAKE) install-dev
endif
	$(PIP_VENV) install build
	$(PYTHON_VENV) -m build

.PHONY: clean-build
clean-build:  ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	$(RM) build dist *.egg-info
	find . -type d -name __pycache__ -exec $(RM) {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

##@ Running
.PHONY: run
run:  ## Run the application with example parameters
	@echo "Running PFM MIP-based Local Search..."
	@echo "Example: Using Taillard instance tai20_5_1.txt with pos_block operator"
	$(PYTHON_VENV) -m $(PROJECT_NAME) \
		--instance tai20_5_1.txt \
		--inst-type taillard \
		--operator pos_block \
		--param-size 10

.PHONY: run-vallada
run-vallada:  ## Run with Vallada instance
	@echo "Running with Vallada instance..."
	$(PYTHON_VENV) -m $(PROJECT_NAME) \
		--instance VFR20_5_1_Gap.txt \
		--inst-type vallada \
		--operator delta \
		--param-size 10