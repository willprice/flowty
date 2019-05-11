Building Flowty
===============

Flowty depends on OpenCV and FFmpeg. To build flowty outside of docker
you will need to build and install these things yourself. This is
surprisingly challenging given the finicky nature of OpenCV builds.
It is also necessary to build OpenCV with CUDA support.

We provide detailed instructions on using conda to build an environment suitable for
flowty. We rely on CUDA 9.2+ due to constraints of the relatively modern compiler
conda-forge use. If you need to use an older CUDA version, then you're best off
compiling everything from scratch (including ffmpeg).

.. contents::


Building with conda
-------------------

.. warning::
    Note that we use ``gcc`` 7.3.0, which is fairly recent and only compatible with
    CUDA 9.2+. If you need to use a previous version, you're on your own and will have to
    build everything from scratch (the conda packages expect a recent C++ ABI and you
    won't be able to compile OpenCV and link to FFmpeg).

The first thing we need to do to produce a CUDA-equipped build of OpenCV is to find out
what graphics card you have, and what driver you're running. This will define what PTX
and BIN files we'll generate when building OpenCV.

.. code-block:: console

    $ nvidia-smi
    Fri May 10 21:33:15 2019
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Quadro K620         Off  | 00000000:01:00.0 Off |                  N/A |
    | 34%   39C    P8     1W /  30W |      1MiB /  2001MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

Now look up the max CUDA version support available for your driver: https://docs.nvidia.com/deploy/cuda-compatibility/index.html
For 418.56, we can use up to CUDA 10.1 (as specified in the top right corner of
``nvidia-smi``'s output,  previous versions of ``nvidia-smi`` don't display this though).

We also see the graphics card is a Quadro K620, we need to look up its CUDA compute
capability here: https://developer.nvidia.com/cuda-gpus. It lists 5.0 compute
capability for this model, so we'll generate CUDA OpenCV binaries for this compute
capability.

Now we need to set up our environment to build OpenCV and flowty. We'll assume you
already have the following installed, if not install them before proceeding:

- ``nvcc`` (CUDA 9.2+, versions earlier than 9.2 don't support GCC 7.3.0, the compiler
  we'll use installed from conda-forge)
- ``conda``


.. hint::

    Any of the following variable definitions with ``<something>`` are placeholders and
    should be replaced based on your setup.

.. code-block:: console

    $ conda create -n flowty -c conda-forge python=3.6 pip ffmpeg lapack openblas \
        cmake libjpeg-turbo libpng zlib cython numpy pytest gxx_linux-64=7.3.0
    $ conda activate flowty
    $ export OPENCV_VERSION=4.1.0
    $ export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib64/pkgconfig
    $ export CC=x86_64-conda_cos6-linux-gnu-gcc CXX=x86_64-conda_cos6-linux-gnu-gcc
    $ export CUDA_PATH="<path-to-cuda>"  # typically is /usr/local/cuda or similar
    $ wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz \
        -O opencv-${OPENCV_VERSION}.tar.gz
    $ wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz \
        -O opencv_contrib-${OPENCV_VERSION}.tar.gz
    $ mkdir -p opencv/build
    $ cd opencv
    $ tar -xvf ../opencv-${OPENCV_VERSION}.tar.gz
    $ tar -xvf ../opencv_contrib-${OPENCV_VERSION}.tar.gz
    $ cd build
    $ unset CXXFLAGS CFLAGS  # conda sets these by default when gxx_linux-64 is installed. These break .cu file compilation
    $ export CUDA_COMPUTE_CAPABILITY=<your-gpu-compute-capability>  # 5.0 in our example.
    $ cmake \
        -D CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
        -D BUILD_opencv_optflow=ON \
        -D BUILD_opencv_cudaoptflow=ON \
        -D BUILD_opencv_tracking=ON \
        -D BUILD_opencv_calib3d=ON \
        -D BUILD_opencv_features2d=ON \
        -D BUILD_opencv_cudafeatures2d=ON \
        -D BUILD_opencv_flann=ON \
        -D BUILD_opencv_plot=ON \
        -D BUILD_opencv_highgui=ON \
        -D WITH_PNG=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_OPENMP=ON \
        -D WITH_OPENCL=ON \
        -D WITH_LAPACK=ON \
        -D WITH_JPEG=ON \
        -D WITH_IPP=ON \
        -D WITH_MKL=ON \
        -D CUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
        -D CUDA_FAST_MATH=ON \
        -D CUDA_ARCH_PTX="$CUDA_COMPUTE_CAPABILITY" \
        -D CUDA_ARCH_BIN="$CUDA_COMPUTE_CAPABILITY" \
        -D ENABLE_FAST_MATH=ON \
        -D WITH_ADE=OFF \
        -D WITH_ARAVIS=OFF \
        -D WITH_CLP=OFF \
        -D WITH_EIGEN=OFF \
        -D WITH_GDAL=OFF \
        -D WITH_GDCM=OFF \
        -D WITH_GPHOTO2=OFF \
        -D WITH_GTK=OFF \
        -D WITH_ITT=OFF \
        -D WITH_JASPER=OFF \
        -D WITH_LIBREALSENSE=OFF \
        -D WITH_MFX=OFF \
        -D WITH_OPENEXR=OFF \
        -D WITH_OPENGL=OFF \
        -D WITH_OPENNI=OFF \
        -D WITH_OPENNI1=OFF \
        -D WITH_OPENVX=OFF \
        -D WITH_PROTOBUF=OFF \
        -D WITH_PTHREADS_PF=OFF \
        -D WITH_PVAPI=OFF \
        -D WITH_QT=OFF \
        -D WITH_QUIRC=OFF \
        -D WITH_TBB=OFF \
        -D WITH_TIFF=OFF \
        -D WITH_V3L=OFF \
        -D WITH_VA=OFF \
        -D WITH_VA_INTEL=OFF \
        -D WITH_VTK=OFF \
        -D WITH_VULKAN=OFF \
        -D WITH_WEBP=OFF \
        -D WITH_XIMEA=OFF \
        -D WITH_XINE=OFF \
        -D WITH_HALIDE=OFF \
        -D WITH_GSTREAMER=OFF \
        -D WITH_V4L=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_opencv_aruco=OFF \
        -D BUILD_opencv_bgsegm=OFF \
        -D BUILD_opencv_bioinspired=OFF \
        -D BUILD_opencv_cudabgsegm=OFF \
        -D BUILD_opencv_cudaobjdetect=OFF \
        -D BUILD_opencv_cudastereo=OFF \
        -D BUILD_opencv_datasets=OFF \
        -D BUILD_opencv_dnn=OFF \
        -D BUILD_opencv_dnn_objdetect=OFF \
        -D BUILD_opencv_dpm=OFF \
        -D BUILD_opencv_face=OFF \
        -D BUILD_opencv_fuzzy=OFF \
        -D BUILD_opencv_gapi=OFF \
        -D BUILD_opencv_hfs=OFF \
        -D BUILD_opencv_img_hash=OFF \
        -D BUILD_opencv_java_bindings_generator=OFF \
        -D BUILD_opencv_js=OFF \
        -D BUILD_opencv_legacy=OFF \
        -D BUILD_opencv_line_descriptor=OFF \
        -D BUILD_opencv_ml=OFF \
        -D BUILD_opencv_objdetect=OFF \
        -D BUILD_opencv_phase_unwrapping=OFF \
        -D BUILD_opencv_photo=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_opencv_python_bindings_generator=OFF \
        -D BUILD_opencv_quality=OFF \
        -D BUILD_opencv_reg=OFF \
        -D BUILD_opencv_rgbd=OFF \
        -D BUILD_opencv_saliency=OFF \
        -D BUILD_opencv_shape=OFF \
        -D BUILD_opencv_stereo=OFF \
        -D BUILD_opencv_stitching=OFF \
        -D BUILD_opencv_stitching=OFF \
        -D BUILD_opencv_structured_light=OFF \
        -D BUILD_opencv_superres=OFF \
        -D BUILD_opencv_surface_matching=OFF \
        -D BUILD_opencv_text=OFF \
        -D BUILD_opencv_videostab=OFF \
        -D BUILD_opencv_xfeatures2d=OFF \
        -D BUILD_opencv_xobjdetect=OFF \
        -D BUILD_opencv_xphoto=OFF \
        ../opencv-${OPENCV_VERSION}

.. note::

    Note that most of the ``cmake`` build options in the above console session are
    disabling additional features of OpenCV unused by flowty; if these cause errors, then
    feel free to drop the ``OFF`` options, they're just to speed up compilation time and save space.


Once configured, you need to double check the ``cmake`` output for FFmpeg, checking it
was found. If you get something like this...

.. code-block:: console

    --   Video I/O:
    --     DC1394:                      NO
    --     FFMPEG:                      NO
    --       avcodec:                   NO
    --       avformat:                  NO
    --       avutil:                    NO
    --       swscale:                   NO
    --       avresample:                NO

... then ``cmake`` has been unable to resolve the location of FFmpeg headers and libs.
You should be aiming for something like this:

.. code-block:: console

    --   Video I/O:
    --     DC1394:                      NO
    --     FFMPEG:                      YES
    --       avcodec:                   YES (58.35.100)
    --       avformat:                  YES (58.20.100)
    --       avutil:                    YES (56.22.100)
    --       swscale:                   YES (5.3.100)
    --       avresample:                YES (4.0.0)

Typically this is as a result of the ``ffmpeg`` pkgconfig files not being installed, or
present within a directory on the ``$PKG_CONFIG_PATH``.

Finally ``make`` and ``make install``:

.. code-block:: console

    $ make -j $(nproc)
    $ make install

Now you should have all the dependencies installed ready to build flowty.

.. code-block:: console

    $ mkdir flowty && cd flowty
    $ export FLOWTY_VERSION=0.0.2
    $ wget https://github.com/willprice/flowty/archive/v${FLOWTY_VERSION}.tar.gz \
        -O flowty.tar.gz
    $ tar -xvf flowty.tar.gz
    $ cd flowty-${FLOWTY_VERSION}
    $ python setup.py build_ext --inplace

You will probably need to add the ``$CONDA_PREFIX/lib64`` directory to your
``$LD_LIBRARY_PATH`` as this is where opencv will have installed its shared libraries.
Without setting this you will get error like...

.. code-block:: console

    ImportError: libopencv_core.so.4 .1: cannot open shared object file: No such file or directory

Resolve this like so:

.. code-block:: console

    $ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64

Check you can run the tests without failures:

.. code-block:: console

    $ PYTHONPATH=src pytest tests

If you were able to run the tests without any import failures, congratulations, you're
now ready to install flowty and compute some flow!

.. code-block:: console

    $ python setup.py install
    $ flowty --help


Building on Ubuntu
------------------

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

Once you've managed to build OpenCV and install it, then have a look at the `flowty
dockerfile <https://github.com/willprice/flowty/blob/master/Dockerfile>`_ to see how
to build and install flowty. Basically it's just:

.. code-block:: console

    $ pip3 install Cython numpy pytest
    $ python setup.py build_ext --inplace
    $ python setup.py install


Resources
---------

For more, check out...

- The `OpenCV compilation docs <https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html>`_
- `CUDA accelerated FFmpeg build <https://gist.github.com/Brainiarc7/988473b79fd5c8f0db54b92ebb47387a>`_

