from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import subprocess
from multiprocessing import cpu_count


def get_cli_output(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd).decode('utf-8')

# libs are of the form -llibname so we drop the first 2 chars to get libname only.
opencv_libs = get_cli_output("pkg-config --libs opencv4").split()
opencv_cflags = get_cli_output("pkg-config --cflags opencv4").split()

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(Extension(
        "*",
        sources=[
            'cv/*.pyx',
        ],
        language='c++',
        include_dirs=[np.get_include()],
        extra_compile_args=opencv_cflags,
        extra_link_args=opencv_libs,
    ), nthreads=cpu_count())
)
