# Thin wrapper around setup.sh so the common commands are one word.
# Run `make help` (or just `make`) to see what's available.

PYTHON := .venv/bin/python3

.PHONY: help setup setup-full setup-train models test run-sim run-webcam record autolabel review export smoke clean

help:
	@echo "make setup       - create .venv, install core deps, run tests"
	@echo "make setup-full  - also install insightface/onnxruntime/pymavlink/pyserial"
	@echo "make models      - download the body re-ID / face ONNX models into models/"
	@echo "make test        - run the test suite (assumes 'make setup' already ran)"
	@echo "make run-sim     - run the toy simulator for 60s"
	@echo "make run-webcam  - webcam dry run with the phone status server"
	@echo "make setup-train - also install PyTorch for training our own models"
	@echo "make record      - record training frames from the webcam (SUBJECT=name optional)"
	@echo "make autolabel   - teacher boxes for new frames"
	@echo "make review      - fix boxes by hand (eval split first)"
	@echo "make export      - build exports/latest for training"
	@echo "make smoke       - prove the training pipeline runs on this machine"
	@echo "make clean       - remove .venv and caches (leaves models/ alone)"

setup:
	./setup.sh

setup-full:
	./setup.sh --full --models

models:
	./setup.sh --models --no-test

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

run-sim:
	$(PYTHON) main.py --platform sim --show --seconds 60

run-webcam:
	$(PYTHON) main.py --platform real --phone --backend opencv --fixed-camera --fixed-pitch 0 --reid fused

setup-train:
	./setup.sh --train --no-test

record:
	$(PYTHON) tools/data.py record $(if $(SUBJECT),--subject $(SUBJECT),)

autolabel:
	$(PYTHON) tools/data.py autolabel

review:
	$(PYTHON) tools/data.py review --split eval

export:
	$(PYTHON) tools/data.py export --out exports/latest

smoke:
	$(PYTHON) -m training.train smoke

clean:
	rm -rf .venv __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
