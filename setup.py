from setuptools import setup, find_packages

setup(name='rudolfpy',
      version='0.1.1',
      packages=find_packages(),
      install_requires = ['numpy',
                          'matplotlib',
                          'numba',
                          'scipy',
                          'tqdm',
      ]
    )