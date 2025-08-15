# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="trading_env_cython",
    ext_modules=cythonize("envs/trading_env_cython.pyx", language_level="3"),
    include_dirs=[numpy.get_include()],
)
