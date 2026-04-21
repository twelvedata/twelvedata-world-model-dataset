PYTHON := .venv/bin/python

.PHONY: install test lint validate backfill update release

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

validate:
	$(PYTHON) scripts/validate_no_leakage.py

backfill:
	$(PYTHON) scripts/backfill.py $(ARGS)

update:
	$(PYTHON) scripts/update_daily.py $(ARGS)

release:
	$(PYTHON) scripts/build_release.py $(ARGS)
