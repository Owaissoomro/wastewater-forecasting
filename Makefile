# Root and tools
ROOT := $(abspath .)
PYTHON ?= python

# Paths
CONFIG_DIR := $(ROOT)/configs
SCRIPTS_DIR := $(ROOT)/scripts
PAPER_DIR := $(ROOT)/paper

# Config files
CONFIG_DEFAULT := $(CONFIG_DIR)/default.yaml
CONFIG_PRIORS := $(CONFIG_DIR)/priors.yaml
CONFIG_LIKELIHOOD := $(CONFIG_DIR)/likelihood.yaml
CONFIG_FORECAST := $(CONFIG_DIR)/forecast.yaml
CONFIG_DETECTION := $(CONFIG_DIR)/detection.yaml
CONFIG_DIAGNOSTICS := $(CONFIG_DIR)/diagnostics.yaml
CONFIG_BENCHMARKS := $(CONFIG_DIR)/benchmarks.yaml
CONFIG_VALIDATION := $(CONFIG_DIR)/validation.yaml

.PHONY: prep priors likelihood forecast detection diagnostics benchmarks validation paper smoke clean

prep:
	$(PYTHON) $(SCRIPTS_DIR)/run_prep.py --config $(CONFIG_DEFAULT)

priors: prep
	$(PYTHON) $(SCRIPTS_DIR)/run_priors.py --config $(CONFIG_DEFAULT) $(CONFIG_PRIORS)

likelihood: prep priors
	$(PYTHON) $(SCRIPTS_DIR)/run_likelihood.py --config $(CONFIG_DEFAULT) $(CONFIG_LIKELIHOOD)

forecast: prep priors likelihood
	$(PYTHON) $(SCRIPTS_DIR)/run_forecast.py --config $(CONFIG_DEFAULT) $(CONFIG_FORECAST)

detection: forecast
	$(PYTHON) $(SCRIPTS_DIR)/run_detection.py --config $(CONFIG_DEFAULT) $(CONFIG_DETECTION)

diagnostics: forecast detection
	$(PYTHON) $(SCRIPTS_DIR)/run_diagnostics.py --config $(CONFIG_DEFAULT) $(CONFIG_DIAGNOSTICS)

benchmarks: likelihood forecast
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmarks.py --config $(CONFIG_DEFAULT) $(CONFIG_BENCHMARKS)

validation: forecast detection
	$(PYTHON) $(SCRIPTS_DIR)/run_validation.py --config $(CONFIG_DEFAULT) $(CONFIG_VALIDATION)

paper: prep priors likelihood forecast detection diagnostics benchmarks validation
	@if [ -f "$(PAPER_DIR)/oxbio_manuscript.tex" ]; then \
		if command -v latexmk >/dev/null 2>&1; then \
			latexmk -pdf -interaction=nonstopmode -halt-on-error -shell-escape -cd "$(PAPER_DIR)/oxbio_manuscript.tex"; \
		else \
			echo "latexmk not found; skipping paper build."; \
		fi; \
	else \
		echo "paper/oxbio_manuscript.tex not found; skipping paper build."; \
	fi

smoke:
	$(PYTHON) $(SCRIPTS_DIR)/smoke.py --config $(CONFIG_DEFAULT)

clean:
	@if [ -d "$(ROOT)/results" ]; then rm -rf "$(ROOT)/results"/*; fi
	@if [ -d "$(PAPER_DIR)" ]; then \
		if command -v latexmk >/dev/null 2>&1; then \
			if [ -f "$(PAPER_DIR)/oxbio_manuscript.tex" ]; then latexmk -C -cd "$(PAPER_DIR)/oxbio_manuscript.tex" || true; fi; \
			if [ -f "$(PAPER_DIR)/appendix.tex" ]; then latexmk -C -cd "$(PAPER_DIR)/appendix.tex" || true; fi; \
		else \
			rm -f "$(PAPER_DIR)"/*.aux "$(PAPER_DIR)"/*.bbl "$(PAPER_DIR)"/*.blg "$(PAPER_DIR)"/*.fdb_latexmk "$(PAPER_DIR)"/*.fls "$(PAPER_DIR)"/*.log "$(PAPER_DIR)"/*.out "$(PAPER_DIR)"/*.toc "$(PAPER_DIR)"/*.synctex.gz; \
		fi; \
	fi