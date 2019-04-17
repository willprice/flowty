from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import subprocess
import os

here = os.path.abspath(os.path.dirname(__file__))


about = {}
with open(os.path.join(here, "flowty", "__version__.py"), "r") as f:
    exec(f.read(), about)


def get_cli_output(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd).decode("utf-8")


# libs are of the form -llibname so we drop the first 2 chars to get
# libname only.
opencv_libs = get_cli_output("pkg-config --libs opencv4").split()
opencv_cflags = get_cli_output("pkg-config --cflags opencv4").split()


def cython_extension(
    pyx_file,
    language="c++",
    include_dirs=None,
    extra_compile_args=None,
    extra_link_args=None,
):
    return Extension(
        pyx_file[: -len(".pyx")].replace("/", "."),
        sources=[pyx_file],
        language="c++",
        extra_compile_args=opencv_cflags,
        extra_link_args=opencv_libs,
    )


extensions = [
    cython_extension(path)
    for path in [
        "flowty/cv/core.pyx",
        "flowty/cv/videoio.pyx",
        "flowty/cv/optflow.pyx",
        "flowty/cv/cuda.pyx",
        "flowty/cv/cuda_optflow.pyx",
        "flowty/cv/cuda_optflow.pyx",
    ]
]

setup(
    name=about["__title__"],
    description=about["__description__"],
    version=about["__version__"],
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions),
    install_requires=["numpy"],
    classifiers=[
        # How mature is this project? Common values
        # are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=[
        "computer-vision",
        "optical-flow",
        "optical",
        "flow",
        "computer",
        "vision",
        "opencv",
        "cython",
        "cuda",
        "gpu",
        "tvl1",
        "farneback",
        "brox",
        "flowty",
    ],
    author=about["__author__"],
    author_email=about["__author_email__"],
    license=about["__license__"],
    url="http://github.com/willprice/flowty",
    project_urls={
        "Bug Tracker": "https://github.com/willprice/flowty/issues",
        "Documentation": "https://flowty.readthedocs.io",
        "Source Code": "http://github.com/willprice/flowty",
    },
    zip_safe=False
)
