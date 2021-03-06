#!/usr/bin/env python

import os
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from skimage._build import cython

    config = Configuration('zmuv', parent_package, top_path)

    cython(['zmuv.pyx'], working_path=base_path)

    config.add_extension(
        'zmuv',
        sources=['zmuv.c'],
        include_dirs=[get_numpy_include_dirs()],
        extra_compile_args=[
            "-fopenmp",
            "-pthread",
            "-O3",
            #"-march=native",
            #"-mtune=native",
            #"-funroll-all-loops",
            #"-fomit-frame-pointer",
            #"-msse4",
            #"-ftree-vectorize",
            #"-ftree-vectorizer-verbose=5",
            #"-ffast-math",
            #"-ftree-loop-distribution",
            #"-funroll-loops",
            #"-ftracer",
            #"-ftree-parallelize-loops=4",
            #"-foptimize-register-move",
            #"-funswitch-loops",
        ],
        extra_link_args=['-fopenmp'],
        )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
