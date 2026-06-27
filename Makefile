.PHONY: help install install-all test lint data gen-synthetic train-baseline train evaluate demo serve autopilot grade report slides clean

help:
	@echo "imgtrans — Document-Image Machine Translation"
	@echo "  install        core install (pip install -e .)"
	@echo "  install-all    full install (pip install -e .[all])"
	@echo "  test           run the test suite (CPU-only, offline)"
	@echo "  data           prefetch/sanity-check datasets (streaming probes)"
	@echo "  gen-synthetic  render synthetic document-image eval pages"
	@echo "  train-baseline persist the dictionary MT baseline"
	@echo "  train          fine-tune the MT core (needs GPU)"
	@echo "  evaluate       MT chrF/BLEU + OCR CER/WER + end-to-end + fidelity"
	@echo "  demo           run the agent on the synthetic seed pages"
	@echo "  serve          start FastAPI + Gradio (/ui)"
	@echo "  autopilot      one-button train->eval->analysis->report+slides+grade"
	@echo "  grade          rubric completeness self-check"

install:
	pip install -e .

install-all:
	pip install -e .[all]

test:
	pytest -q

lint:
	ruff check src tests || true

data:
	imgtrans data

gen-synthetic:
	imgtrans gen-synthetic

train-baseline:
	imgtrans train-baseline

train:
	imgtrans train-mt

evaluate:
	imgtrans evaluate --fast

demo:
	imgtrans demo-agent --fast

serve:
	imgtrans serve --ui

autopilot:
	imgtrans autopilot --no-train

grade:
	imgtrans grade

report:
	imgtrans generate-report

slides:
	imgtrans generate-slides

clean:
	rm -rf build dist *.egg-info .pytest_cache src/imgtrans/__pycache__
