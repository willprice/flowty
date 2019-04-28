Building Flowty
===============

Flowty depends on OpenCV and FFmpeg. To build flowty outside of docker
you will need to build and install these things yourself. This is
surprisingly challenging given the finicky nature of OpenCV builds.
It is also necessary to build OpenCV with CUDA support.

Your best bet for manually setting up an environment to run flowty is to look
at the Dockerfile we build upon: `willprice/opencv4 <https://github.com/dl-container-registry/opencv4/blob/master/Dockerfile>`_.
This is built on Ubuntu 18.04 (although very few changes are needed to go back to 16.04).
You can see the flags we enable for building OpenCV. The key flags are:

- ``OPENCV_GENERATE_PKGCONFIG=on`` as we use ``pkgconfig`` in ``setup.py`` to get the OpenCV paths.
- ``WITH_FFMPEG=on`` since FFmpeg is the default backend
- ``WITH_CUDA=on`` as some of the algorithms are CUDA accelerated
- ``OPENCV_EXTRA_MODULES_PATH=<path/to/opencv_contrib/modules>`` since the ``optflow`` module isn't in core OpenCV.

Pay close attention to the output of ``cmake`` as the configuration step won't
crash if FFmpeg isn't found, this will result in an OpenCV build incapable of
reading videos (upon attempting to read a video it will just return no frames
rather than raising an exception).


Resources
---------

Check out the following 

- The `FFmpeg compilation guide <https://www.google.com/search?client=firefox-b-d&q=ffmpeg+build>`_
- The `OpenCV compilation docs <https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html>`_
- `CUDA accelerated FFmpeg build <https://gist.github.com/Brainiarc7/988473b79fd5c8f0db54b92ebb47387a>`_

