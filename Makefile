SHELL := bash
BASE_NAME := flow
CONTAINER_NAME := willprice/$(BASE_NAME)
SINGULARITY_NAME := $(BASE_NAME).simg
TAG := cuda-10.1-cudnn7
SRC := $(shell find . -iregex '.*\.(pyx|pxd|py)')

.PHONY: all
all: build singularity

.PHONY: build
build: $(SRC)
	CC=clang python setup.py build_ext --inplace

install:


.PHONY: build
docker_build:
	docker build -t $(CONTAINER_NAME):$(TAG) .

.PHONY: push
docker_push:
	docker push $(CONTAINER_NAME):$(TAG)

.PHONY: singularity
singularity_build: $(SINGULARITY_NAME)

.PHONY: clean
clean:
	@rm -rf build dist flowty/cv/*.{cpp,c,so} *.egg-info
	@rm -rf `find . -iname '__pycache__' -o -iname '*.pyc'` 

.PHONY: check
check: build
	pytest tests

.PHONY: test
test: check


$(SINGULARITY_NAME):
	singularity build $@ Singularity

