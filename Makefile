PYTHON ?= python

.PHONY: smoke-test test

smoke-test:
	$(PYTHON) pipeline/run_smoke_test.py

test:
	$(PYTHON) -m pytest
