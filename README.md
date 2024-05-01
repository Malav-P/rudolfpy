# `rudolfpy`

Generic filter implementations in python, named after Rufolf E. Kálmán himself. 

Note: [`FilterPy`](https://filterpy.readthedocs.io/en/latest/) is a "pedalogical" tool, and may compromise on performance. In contrast, the filters in this repository are meant to be performant "enough" for the purpose of doing research.


### Dependencies:

- numpy, matplotlib, numba, scipy, tqdm


## Quick start

The basic usage of `rudolfpy` is:

1. Define dynamics object (or use one of the implemented models within `rudolfpy`)
2. Define measurement object (or use one of the implemented models within `rudolfpy`)
3. Define filter object
4. Construct `rudolfpy.Recursor()` object
5. Run recursion, either via `rudolfpy.Recursor.recurse_measurements_list()` or via `rudolfpy.Recursor.recurse_measurements_func()`

For examples, see: `dev/dev_ekf_recurse_func.py` or `dev/dev_ekf_recurse_list.py`.