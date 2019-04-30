from .__version__ import *
from .cv import cuda

try:
    cv.cuda.get_device()
    cuda_available = True
except RuntimeError:
    cuda_available = False

