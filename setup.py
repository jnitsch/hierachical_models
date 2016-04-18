#!/usr/bin/env python

from distutils.core import setup

setup(name='hierachical_models',
      version='0.0',
      description='Tool for comparing different classifiers',
      author='Julia Nitsch',
      author_email='julia.nitsch@gmail.com',
      license='BSD',
      packages=['hierachical_models',
                'hierachical_models.io',
                'hierachical_models.classifiers',
                'hierachical_models.metrics',],
      package_dir={'': 'src'},
      scripts=[],
      install_requires=['h5py'],
      zip_safe=False)


