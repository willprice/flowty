# Flowty
![Build status](https://img.shields.io/circleci/project/github/willprice/flowty/master.svg)
[![Dockerhub](https://img.shields.io/badge/docker-flowty-informational.svg)](https://cloud.docker.com/repository/docker/willprice/flowty)
[![Docker image size](https://images.microbadger.com/badges/image/willprice/flowty.svg)](https://microbadger.com/images/willprice/flowty)

Flowty is the swiss army knife of computing optical flow. Flowty is...

- Performant—leveraging CUDA accelerated optical flow implementations.
- Easy of use—packaged in docker so you don't have to compile OpenCV and 
  Flowty yourself.

## Usage

Visit https://flowty.rtfd.org to learn more about how to obtain and use flowty.
In a nutshell:

```sh
$ ls /path/to/media
video.mp4

$ docker run --rm --runtime=nvidia willprice/flowty

[Flowty help description]
  
$ docker run -it --rm \
  --runtime=nvidia \
  --mount type=bind,source=/path/to/media,target=/data \
  willprice/flowty /data/video.mp4 /data/flow/{axis}/{index:05d}.jpg --cuda tvl1
```
