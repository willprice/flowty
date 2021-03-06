SHELL := bash
BASE_NAME := flowty
CONTAINER_NAME := willprice/$(BASE_NAME)
SINGULARITY_NAME := $(BASE_NAME).simg
TAG := latest
SRC := $(shell find . -iregex '.*\.(pyx|pxd|py)')
PYTHON := python3

.PHONY: all
all: build singularity

.PHONY: build
build: $(SRC)
	$(PYTHON) setup.py build_ext --inplace

.PHONY: install
install:
	$(PYTHON) setup.py install

.PHONY: venv
venv: .venv
.venv:
	$(PYTHON) -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: check test
test: check
check: build
	$(PYTHON) -m pytest tests

.PHONY: docs
docs: 
	$(MAKE) -C docs html

.PHONY: build
docker_build:
	docker build -t $(CONTAINER_NAME):$(TAG) .

.PHONY: push
docker_push:
	docker push $(CONTAINER_NAME):$(TAG)

.PHONY: singularity
singularity_build: $(SINGULARITY_NAME)
$(SINGULARITY_NAME):
	singularity build $@ Singularity


.PHONY: clean
clean:
	@rm -rf build dist src/flowty/cv/*.{cpp,c,so} *.egg-info
	@rm -rf `find . -iname '__pycache__' -o -iname '*.pyc'` 
