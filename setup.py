from setuptools import setup, find_packages

setup(name='rudolfpy',
      version='1.0.0',
      packages=find_packages(),
      install_requires = ['numpy',
                          'matplotlib',
                          'numba',
                          'scipy',
                          'tqdm',
      ]
    )