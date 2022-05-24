import py_newton
import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.cfunc("double(double,double)")
def add(a, b):
    return a + b

py_newton.get_func(add)

N, a, b, c, d = 100, -1, 1, -1, 1
py_newton.newton_meshgrid(100, -1, 1, -1, 1)

grid = np.array(np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N)))
iters = np.zeros_like(grid[0, ...])
A = np.zeros_like(grid[0, ...])
res = py_newton.newton_grid(*grid, A, iters)