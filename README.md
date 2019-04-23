# Flowty

**WARNING**

**Flowty is a work in progress, don't expect anything to work yet!**

Flowty is a tool for computing optical flow. It's goals are to be performant,
flexible, and above all, easy to use.

## Usage

Since Flowty is built upon OpenCV and it's CUDA extensions, it is necessary to
have a working installation of these if you wish to use Flowty in your
environment, for this reason, we recommend use our precompiled docker images

### Prerequisites:

* [docker-ce](https://docs.docker.com/install/)
* [nvidia-docker 2.0+](https://github.com/NVIDIA/nvidia-docker)

### Running Flowty

```sh
$ ls /path/to/media
video.mp4

$ docker run --runtime=nvidia willprice/flowty

[Flowty help description]

$ docker run -it --rm \
  --runtime=nvidia \
  --mount type=bind,source=/path/to/media,target=/data \
  willprice/flowty /data/video.mp4 /data/flow/{axis}/{index:05d}.jpg
```

## Contributing

### Adding a new method for computing optical flow

We'd love for you to add additional methods for computing optical flow. We've
designed Flowty to facilitate this.
