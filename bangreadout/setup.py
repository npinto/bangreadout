#!/usr/bin/env python

import os
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from skimage._build import cython

    config = Configuration('fast_zmuv', parent_package, top_path)
    #config.add_data_dir('tests')

    cython(['fast_zmuv.pyx'], working_path=base_path)

    config.add_extension(
        'fast_zmuv',
        sources=['fast_zmuv.c'],
        include_dirs=[get_numpy_include_dirs()],
        extra_compile_args=[
            "-fopenmp",
            "-pthread",
            "-O3"
        ],
        extra_link_args=['-fopenmp'],
        )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
