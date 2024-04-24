from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  # Make sure NumPy is installed

extensions = [
    Extension("cython_cb_svdpp", ["cython_cb_svdpp.pyx"],
              include_dirs=[np.get_include()])  # Adding NumPy headers
]

setup(
    name='CB_SVDpp',
    ext_modules=cythonize(extensions),
    zip_safe=False
)
