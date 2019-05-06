=================
Welcome to Flowty
=================

.. toctree::
   :hidden:
   :maxdepth: 3

   index
   io_formats
   build
   development

Flowty (pronounced `floʊtiː <https://en.wikipedia.org/wiki/Help:IPA/English>`_) is the swiss army knife of computing `optical flow <https://en.wikipedia.org/wiki/Optical_flow>`_.


Flow methods
============

The following OpenCV optical flow methods are implemented:

- `TV-L1 <https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22>`_
  (OpenCV reference
  `CPU\ <https://docs.opencv.org/4.1.0/dc/d4d/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html>`_
  / `GPU\ <https://docs.opencv.org/4.1.0/d6/d39/classcv_1_1cuda_1_1OpticalFlowDual__TVL1.html>`_)
- `Brox <https://lmb.informatik.uni-freiburg.de/people/brox/pub/brox_eccv04_of.pdf>`_ (OpenCV reference `GPU\ <https://docs.opencv.org/4.1.0/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html>`_)
- `Pyramidal Lucas-Kanade <http://robots.stanford.edu/cs223b04/algo_affine_tracking.pdf>`_ (OpenCV reference `GPU\ <https://docs.opencv.org/4.1.0/d0/da4/classcv_1_1cuda_1_1DensePyrLKOpticalFlow.html>`_)
- `Farneback <http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf>`_ (OpenCV reference `CPU\ <https://docs.opencv.org/4.1.0/de/d9e/classcv_1_1FarnebackOpticalFlow.html>`_ / `GPU\ <https://docs.opencv.org/4.1.0/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html>`_)
- `Dense Inverse Search <https://arxiv.org/abs/1603.03590>`_ (OpenCV reference `CPU\ <https://docs.opencv.org/4.1.0/de/d4f/classcv_1_1DISOpticalFlow.html>`_)
- `Variational Refinement <https://lmb.informatik.uni-freiburg.de/people/brox/pub/brox_eccv04_of.pdf>`_ (OpenCV reference `CPU\ <https://docs.opencv.org/4.1.0/d2/d4b/classcv_1_1VariationalRefinement.html>`_)

Roadmap
-------

The following methods aren't implemented, but are on the roadmap to implement next.

- `PCA flow <http://files.is.tue.mpg.de/black/papers/cvpr2015_pcaflow.pdf>`_ (OpenCV reference `CPU\ <https://docs.opencv.org/4.1.0/d1/da2/classcv_1_1optflow_1_1OpticalFlowPCAFlow.html>`_)
- `Robust local optical flow <http://elvera.nue.tu-berlin.de/files/1498Geistert2016.pdf>`_ (OpenCV reference `CPU\ <https://docs.opencv.org/4.1.0/df/d59/classcv_1_1optflow_1_1DenseRLOFOpticalFlow.html>`_)


Usage
=====

Flowty is packaged as a 
`docker container <https://cloud.docker.com/repository/docker/willprice/flowty>`_ 
to save you the hassle of having to build an accelerated version of OpenCV
linked with FFmpeg by hand.

Dependencies
------------

You will need the following software installed on your host:

- `docker-ce <https://docs.docker.com/install/>`_
- `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_

To check these prerequisites are satified, run

.. code-block:: console

    $ docker run --runtime=nvidia --rm nvidia/cuda:10.1-base nvidia-smi

It should print the output of ``nvidia-smi``

Invoking flowty
---------------

The container will run ``flowty`` by default, so you can provide arguments like you would if you were running ``flowty`` installed natively on your host.

For example, to compute TV-L1 optical flow from the video `/absolute/path/to/video_dir/video.mp4` and save the flow u, v components as separate JPEGs in `/absolute/path/to/video_dir/flow/{axis}/`, run the following command:

.. code-block:: console

   # CPU only version
   $ docker run --rm -it \
       --mount "type=bind,source=/absolute/path/to/video_dir,target=/data" \
       willprice/flowty \
       tvl1 "/data/video.mp4" "/data/flow/{axis}/frame_{index:05d}.jpg"

   # GPU accelerated version
   $ docker run --runtime=nvidia --rm -it \
       --mount "type=bind,source=/absolute/path/to/video_dir,target=/data" \
       willprice/flowty \
       tvl1 "/data/video.mp4" "/data/flow/{axis}/frame_{index:05d}.jpg" --cuda

Check out :doc:`io_formats` to find out more about supported formats.

Explanation
-----------

As docker isn't heavily used in the computer vision community we'll break
the above example command down piece by piece to explain what's going on.

First, an explanation of what Docker is: Docker is a container platform,
it allows you to build and run containers, which are a little like
light-weight VMs. A container contains an entire OS and all the
dependencies of the application you'd like to run. It doesn't share any
storage with the host by default, so if you want to access files on your
computer from inside the container you need to *mount* the directory
into the container, this allows you to read and write to a host
directory from within the container.

- ``docker run`` is used to run an *instance* of a container, this is
  the equivalent of launching a VM, but is much quicker.
- ``--rm`` indicates that we want to remove the container after usage
  (to prevent it taking up space)
- ``-it`` is short for ``--interactive`` and ``--tty``.
  ``--interactive`` hooks up STDIN from your console to the container,
  allowing you to Ctrl-C to kill the operation, ``--tty`` allocates a
  pseudo-TTY and pipes this to STDOUT so that you can see the log
  messages printed by the tool, if this wasn't present no output from
  the container would be printed.
- ``--runtime=nvidia`` is used to switch out the docker container
  backend to NVIDIA's version which injects the necessary hooks to run
  CUDA programs. It is only necessary to specify this if you are using
  the CUDA accelerated routines (i.e. you pass the ``--cuda`` arg to
  flowty)
- ``--mount type=bind,source=/absolute/path/to/video_dir,dest=/data`` is
  the specification to docker that allows you to access a host directory
  within the container, this is called a *bind* mount, hence the
  ``type=bind`` specification. We mount the directory
  ``/absolute/path/to/video_dir`` on the host to ``/data`` within the
  container. Now anytime the container writes or reads anything from
  ``/data`` it will actually read from ``/absolute/path/to/video_dir/``.
