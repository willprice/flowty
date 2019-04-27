Building Flowty
===============

Flowty depends on OpenCV and FFmpeg. To build flowty outside of docker
you will need to build and install these things yourself. This is
surprisingly challenging given the finicky nature of OpenCV builds.
It is also necessary to build OpenCV with CUDA support.


Resources
---------

Check out the following 

- The `FFmpeg compilation guide <https://www.google.com/search?client=firefox-b-d&q=ffmpeg+build>`_
- The `OpenCV compilation docs <https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html>`_
- `CUDA accelerated FFmpeg build <https://gist.github.com/Brainiarc7/988473b79fd5c8f0db54b92ebb47387a>`_
