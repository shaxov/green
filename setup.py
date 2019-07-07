from distutils.core import setup, Extension
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

# define the extension module
green_module = Extension('green', sources=['green_module.c'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=[])

# run the setup
setup(ext_modules=[green_module])
