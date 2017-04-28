from distutils.core import setup, Extension
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name = 'Solver',
  ext_modules=[
    Extension('solve',
              sources=['solve.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['/GS /GL /analyze- /W3 /Gy /Zc:wchar_t /Zi /Gm- /O2 /sdl /Arch:AVX2'],
              language='c++')
    ],
  cmdclass = {'build_ext': build_ext}
)
