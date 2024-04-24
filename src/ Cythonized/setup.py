from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("cb_svdpp.pyx"),
    include_dirs=[np.get_include()]
)
