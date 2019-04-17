#!/usr/bin/env python3


import matplotlib.pyplot as plt
from videoio import VideoSource
from optflow import TvL1OpticalFlow
import numpy as np

src = VideoSource('mr-bubz.mp4')
frame1 = next(src)

frame2 = next(src)

alg = TvL1OpticalFlow()
flow = alg(frame1, frame2)
np_frame = flow.asarray()
print(type(np_frame), np_frame.dtype, np_frame.shape)
plt.imshow(np_frame[:, :, 0])
plt.show()
plt.imshow(np_frame[:, :, 1])
plt.show()

# vim: set ft=python:

