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

.PHONY: test
test: build
	PYTHONPATH=. pytest tests

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
	@rm -rf build cv/*.{cpp,c}
$(SINGULARITY_NAME):
	singularity build $@ Singularity
