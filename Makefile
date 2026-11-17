# Makefile — Bare-Metal sEMG Classification & Quantized 1D-CNN QEMU Suite
SHELL := /bin/bash
PYTHON := .venv/bin/python

.PHONY: all train_cnn export_cnn test_qemu_cnn report_cnn test_qemu_freertos report_freertos clean

all: test_qemu_cnn test_qemu_freertos

train_cnn:
	$(PYTHON) src/models/train_cnn1d_quantized.py

export_cnn:
	$(PYTHON) src/export/export_cnn1d_c_header.py

test_qemu_cnn:
	@if [ ! -f embedded/firmware/include/cnn1d_weights_int8.h ]; then \
		$(PYTHON) src/models/train_cnn1d_quantized.py && \
		$(PYTHON) src/export/export_cnn1d_c_header.py; \
	fi
	$(PYTHON) scripts/run_qemu_cnn1d_benchmark.py
	$(PYTHON) scripts/generate_cnn1d_vs_lda_report.py

report_cnn:
	$(PYTHON) scripts/generate_cnn1d_vs_lda_report.py

test_qemu_freertos:
	$(PYTHON) scripts/run_qemu_freertos_dualcore_benchmark.py
	$(PYTHON) scripts/generate_freertos_dualcore_report.py

report_freertos:
	$(PYTHON) scripts/generate_freertos_dualcore_report.py

clean:
	rm -rf embedded/firmware/build
