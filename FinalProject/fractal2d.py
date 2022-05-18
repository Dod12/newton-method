import itertools
from logging import warning, info
from numbers import Real
from typing import Callable

import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm


class Fractal2D(object):

    def __init__(
            self,
            function: Callable[[NDArray[Real]], NDArray[Real]],
            jacobian: Callable[[NDArray[Real]], NDArray[Real]] = None,
            compile: bool = True,
    ) -> None:

        try:
            isinstance(function(np.zeros((2,))), np.ndarray)
            if compile is True:
                function = numba.jit(function, nopython=True, parallel=True)
            if jacobian is not None:
                isinstance(jacobian(np.zeros((2,))), np.ndarray)
                if compile is True:
                    jacobian = numba.jit(jacobian, nopython=True, parallel=True)
        except TypeError as e:
            raise TypeError(f"Function must take a tuple of two numbers and return a single real. Got: {e}")
        else:
            self.function = function
            self.jacobian = jacobian
            self.zeroes = list()

    def newton_zeros(
            self, x_0: NDArray[Real], n_iter: int = 1000, h: float = 1e-5,
            loop_tolerance: float = np.finfo(np.float64).eps, comparison_tolerance: float = 1e-9
    ) -> NDArray[np.float64]:
        if self.jacobian is not None:
            x_n = self._newton_helper(self.function, self.jacobian, x_0, n_iter, loop_tolerance)
        else:
            x_n = self._newton_helper_estimate_jacobian(self.function, x_0, n_iter, h, loop_tolerance)

        if np.any(np.isnan(x_n)) or not np.linalg.norm(self.function(x_n)) < comparison_tolerance:
            warning(RuntimeWarning(f"Newton's method on {x_0} failed to converge in {n_iter} iterations."))
            return np.array((np.nan, np.nan))
        else:
            info(f"Estimated root of {x_0} to {x_n}.")
            return x_n

    def newton_index(self, x0: NDArray[Real], tol: float = 1e-9, **kwargs) -> int:
        root = self.newton_zeros(x0, **kwargs, comparison_tolerance=tol)
        for i, item in enumerate(self.zeroes):
            if np.linalg.norm(item - root) < tol or (np.any(np.isnan(item)) and np.any(np.isnan(root))):
                return i
        self.zeroes.append(root)
        return len(self.zeroes) - 1

    def plot(self, N, a, b, c, d, n_cpus=1):

        grid = np.array(np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N)))
        self.A = np.zeros_like(grid[0, ...])
        for i, j in tqdm(itertools.product(range(grid.shape[1]), range(grid.shape[2])), total=N ** 2):
            self.A[i, j] = self.newton_index(grid[:, i, j])
        self.mesh = plt.imshow(self.A, extent=[a, b, c, d], origin="lower", aspect="equal", interpolation=None)
        plt.show(block=True)

    @staticmethod
    @numba.jit(nopython=True)
    def _newton_helper(function: Callable, jacobian: Callable, x_0: NDArray[Real], n_iter: int = 10000,
                       loop_tolerance: float = np.finfo(np.float64).eps) -> NDArray[np.float64]:
        iterations = 0
        x_n = x_0

        while not np.linalg.norm(function(x_n)) < loop_tolerance and iterations < n_iter:
            if not (np.any(np.isfinite(x_n)) and np.any(np.isfinite(jacobian(x_n))) and np.any(
                    np.isfinite(function(x_n)))) or np.linalg.det(jacobian(x_n)) == 0:
                break
            x_n = x_n - np.linalg.inv(jacobian(x_n)) @ function(x_n)
            iterations += 1
        return x_n

    @staticmethod
    @numba.jit(nopython=True)
    def _newton_helper_estimate_jacobian(function: Callable, x_0: NDArray[Real], n_iter: int = 10000,
                                         h: float = 1e-3, loop_tolerance: float = np.finfo(np.float64).eps) -> NDArray[
        np.float64]:
        iterations = 0
        x_n = x_0

        while not np.linalg.norm(function(x_n)) < loop_tolerance and iterations < n_iter:
            # TODO: Check for divergence
            if not (np.any(np.isfinite(x_n)) and np.any(np.isfinite(function(x_n)))):
                break
            # Let's estimate the jacobian matrix
            df1_dx1, df2_dx1 = (function(x_n + np.array((h, 0))) - function(x_n - np.array((h, 0)))) / (2 * h)
            df1_dx2, df2_dx2 = (function(x_n + np.array((0, h))) - function(x_n - np.array((0, h)))) / (2 * h)
            jacobian = np.array([[df1_dx1, df1_dx2],
                                 [df2_dx1, df2_dx2]])

            # If the jacobian is singular, Newton's method fails to converge
            if not np.any(np.isfinite(jacobian)) or np.linalg.det(jacobian) == 0:
                break

            x_n = x_n - np.linalg.inv(jacobian) @ function(x_n)
            iterations += 1
        return x_n
