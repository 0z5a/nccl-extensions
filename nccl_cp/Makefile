SHELL := /bin/sh
.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CUDA_HOME ?= /usr/local/cuda
JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
export CUDA_HOME

.PHONY: help nccl-submodule
help:
	@printf '%s\n' 'make nccl-submodule  Initialize and build the pinned NCCL dependency' \
	  '                     Overrides: CUDA_HOME=/path/to/cuda JOBS=8' \
	  'cmake -S . -B build  Configure NCCL CP' \
	  'cmake --build build  Build NCCL CP'

nccl-submodule:
	$(MAKE) -C "$(PROJECT_ROOT)/.." nccl-submodule
