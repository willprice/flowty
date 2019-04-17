# cython: language_level = 3

from .c_cuda cimport getDevice, getCudaEnabledDeviceCount, printCudaDeviceInfo, printShortCudaDeviceInfo, resetDevice, setDevice

def get_device():
    return getDevice()

def get_cuda_enabled_device_count():
    return getCudaEnabledDeviceCount()

def print_cuda_device_info(int device=get_device()):
    printCudaDeviceInfo(device)

def print_short_cuda_device_info(int device=get_device()):
    printShortCudaDeviceInfo(device)

def reset_device():
    resetDevice()

def set_device(int device):
    setDevice(device)
