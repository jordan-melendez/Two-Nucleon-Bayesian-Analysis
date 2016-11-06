from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import numpy
# from Cython.Compiler.Options import directive_defaults
Cython.Compiler.Options.annotate = True
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# Run with: python create_cython.py build_ext -i

ext_modules = [
    Extension(name='matrix_operations',
              sources=['matrix_operations.pyx'],
              extra_compile_args=["-O2", "-ffast-math"],
              libraries=["gsl", "gslcblas", "m"],
              include_dirs=[numpy.get_include()]
              # define_macros=[('CYTHON_TRACE', '1')]
              ),
]

setup(
    name='Jordan_test',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
