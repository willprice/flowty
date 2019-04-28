Input/Output Formats
====================

Input
-----

Flowty uses the OpenCV `VideoCapture <https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html>`_ API to load videos. It defaults to the FFmpeg backend.

Flowty is built to support reading from:

- H263
- H264
- VP9
- JPEG

For any of the video formats you can simply provide ``/path/to/video.ext`` as
the ``src`` argument.  To use a folder of sequentially numbered JPEGs as input
you have to provide an image filename template such as
``/path/to/video_dir/frame_%06d.jpg``
(like specified in the ``VideoCapture`` 
`constructor docs <https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html#ac4107fb146a762454a8a87715d9b7c96>`_).


Output
------

Currently only one output format is supported: splitting flow into u, v pairs,
quantising them and storing them as JPEGs. The ``dest`` argument must be a `python 
format template <https://docs.python.org/3.6/library/string.html#formatspec>`_ 
with two interpolations: ``{axis}`` and ``{index}``.

- ``{axis}`` will be replaced with ``u`` or ``v`` depending upon the direction of the flow field being written.
- ``{index}`` will be replaced by the flow frame index being written, typically you'll want to add some format
  specifiers to this to 0 pad it, e.g. ``{index:06d}``.

An example template ``dest`` string is ``/data/flow/{axis}/frame_{index:06d}.jpg``.
