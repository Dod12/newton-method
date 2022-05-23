import itertools
from logging import warning, info
from numbers import Real
from typing import Callable, Union, Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.typing import NDArray
import tqdm.auto as tqdm


class Fractal2D(object):

    def __init__(
            self,
            function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
            jacobian: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = None,
            compile: bool = True, n_iter: int = 10000, h: float = 1e-5,
            loop_tolerance: float = np.finfo(np.float64).eps * 1e3,
            comp_tolerance: float = 1e-9, simplified: bool = False
    ) -> None:
        """
        Class for creating fractal plots based on Newton's method.

        This class acts as a container for functions and can be used to estimate the zeros of a function R^2 -> R^2.
        The function and it's derivative (optionally) are passed to the constructor and can be compiled just in time for
        greater speed.

        Parameters
        ----------
        function : Callable[[NDArray[Real]], NDArray[Real]]
            Function to use for estimation of zeros. This is a python function that takes a numpy array and returns a
            numpy array.
        jacobian : Callable[[NDArray[Real]], NDArray[Real]], optional
            Jacobian for the function. This should be a python function taking a 2x1 array and return a 2x2 array of the
            partial derivatives of the function. The default is None, estimating the jacobian using the finite
             differences method.
        compile : bool, optional
            Whether to compile the functions defining the function and jacobian. This leads to a massive speedup of up
            to 1000 times. The default is True.
        n_iter : int, optional
            Maximum number of iterations to run before returning the zero. Increasing this can yield better accuracy
            for some functions. The default is 10000.
        h : float, optional
            Finite difference to use when estimating jacobian. Defaults to 1e-5, providing good accuracy and robustness
            to floating-point errors. Only used when jacobian is None.
        loop_tolerance : float, optional
            Tolerance for determining convergence of Newton's method. The default is machine epsilon (usually ≈ 1e-16).
        comp_tolerance : float, optional
            Tolerance for determining whether a point is a zero. The default is 1e-9.
        simplified : bool, optional
            Whether to use the simplified Newton's method. Default is False, not using simplified method. See Notes for
            descriptions of what simplified method entails.

        Raises
        ------
        TypeError
            If the function or jacobian can't compute some simple test cases.

        Returns
        -------
        Fractal2D :
            self
        """
        if compile is True:
            function = numba.jit(function, nopython=True, nogil=True, fastmath=True)
            if jacobian is not None:
                jacobian = numba.jit(jacobian, nopython=True, nogil=True, fastmath=True)

        out = function(np.zeros((2,)), np.zeros((2,)))
        if not isinstance(out, np.ndarray):
            raise TypeError(f"Function must return a numpy array of shape (2,), not {type(out)}.")
        elif out.shape != (2,):
            raise TypeError(f"Function must return a numpy array of shape (2,), not {out.shape}.")

        if jacobian is not None:
            out = jacobian(np.zeros((2,)), np.zeros((2, 2)))
            if not isinstance(out, np.ndarray):
                raise TypeError(f"Jacobian must return a numpy array of shape (2,), not {type(out)}.")
            elif out.shape != (2, 2):
                raise TypeError(f"Jacobian must return a numpy array of shape (2,2), not {out.shape}.")
        self.function = function
        self.jacobian = jacobian
        self.zeros = list()
        self.n_iter = n_iter
        self.h = h
        self.loop_tolerance = loop_tolerance
        self.comp_tolerance = comp_tolerance
        self.simplified = simplified
        self.A = None
        self.iters = None
        self.roots = None

    def plot(self, N: int, a: float, b: float, c: float, d: float):
        """
        This method plots the fractal

        Parameters
        ----------
        N : int
            Resolution of image
        a : float
            starting point of x axes
        b : float
            end point of x axes
        c : float
            starting point of y  axes
        d : float
            ending point of y axes

        Returns
        -------
        None.

        """

        # Define the grid of points to calculate and transform them to an array.
        grid = np.array(np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N)))
        self.roots = np.zeros_like(grid)
        self.iters = np.zeros_like(grid[0, ...])
        self.A = np.zeros_like(grid[0, ...])
        self.roots, self.iters = self._loop_helper(self._newton_helper, self.function, grid, jacobian=self.jacobian,
                                                   n_iter=self.n_iter, h=self.h, loop_tolerance=self.loop_tolerance,
                                                   comp_tolerance=self.comp_tolerance, simplified=self.simplified)

        self.A = self.newton_index(self.roots)
        self.mesh = plt.imshow(self.A, extent=[a, b, c, d], origin="lower", aspect="equal", interpolation=None)
        plt.show(block=True)

        # ToDo: add dependance on Simplified, make some fcns to make a simplified newton method.

    def newton_index(self, roots: NDArray[Real]) -> NDArray[np.int64]:

        indexes = np.zeros(roots.shape[1:])

        for i, j in tqdm.tqdm(itertools.product(range(roots.shape[1]), range(roots.shape[2])),
                              total=roots.shape[1] * roots.shape[2], smoothing=0, desc="Calculating indices"):
            root = roots[:, i, j]
            for k, zero in enumerate(self.zeros):
                if np.allclose(root, zero, equal_nan=True):
                    indexes[i, j] = k
                    break
            else:
                self.zeros.append(root)
                indexes[i, j] = len(self.zeros) - 1
        return indexes

    @staticmethod
    def _loop_helper(ufunc: Callable, function: Union[Callable, None], X: NDArray[Real],
                     jacobian: Union[Callable, None] = None, n_iter: int = 10000, h: float = 1e-5,
                     loop_tolerance: float = np.finfo(np.float64).eps, comp_tolerance: float = 1e-9,
                     simplified: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Helper function to call a function over an array of elements using parallelism.

        Parameters
        ----------
        ufunc : Callable
            Function to call
        X : NDArray[Real]
            Array to iterate over the last two dimensions.
        args
            Positional arguments to `func`.
        kwargs
            Keyword arguments to `func`.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.int64]] :
            Array of results.
        """

        zeros = np.zeros_like(X)
        iters = np.zeros_like(X[0,...])
        for i, j in tqdm.tqdm(itertools.product(range(X.shape[1]), range(X.shape[2])),
                              total=X.shape[1] * X.shape[2], smoothing=0, desc="Computing zeros"):
            zeros[:, i, j], iters[i, j] = ufunc(function, X[:, i, j], jacobian, n_iter, h, loop_tolerance,
                                                comp_tolerance, simplified)
        return zeros, iters

    @staticmethod
    def _newton_helper(function: Callable, x_0: NDArray[np.float64], jacobian: Union[Callable, None] = None,
                       n_iter: int = 10000, h: float = 1e-5, loop_tolerance: float = np.finfo(np.float64).eps/2,
                       comp_tolerance: float = 1e-9, simplified: bool = False) -> Tuple[
        NDArray[np.float64], NDArray[np.int64]]:
        """
        Implement Newton's method to estimate zeros.

        Notes
        -----
        This function treats four different cases separately. The simplified method computes and inverses the jacobian
        once for the initial value and uses this for Newton's method. This brings a speedup because estimating and
        inverting the jacobian is very expensive computationally. If the jacobian is None, the jacobian matrix is
        estimated using the finite differences method.

        Parameters
        ----------
        function : Callable
            Function to estimate zeros for.
        x_0 : NDArray[Real]
            Initial guess of the zero-point to start estimation from.
        jacobian : Callable, optional
            Function defining the jacobian matrix for the function we wish to estimate zeros for. Defaults to None,
            using finite differences to estimate the jacobian matrix.
        n_iter : int, optional
            Maximum number of iterations to run before returning the zero. Increasing this can yield better accuracy
            for some functions. The default is 10000.
        h : float, optional
            Finite difference to use when estimating jacobian. Defaults to 1e-5, providing good accuracy and robustness
            to floating-point errors. Only used when jacobian is None.
        loop_tolerance : float, optional
            Tolerance for determining convergence of Newton's method. The default is machine epsilon (usually ≈ 1e-16).
        comp_tolerance : float, optional
            Tolerance for determining whether a point is a zero. The default is 1e-9.
        simplified : bool, optional
            Whether to use the simplified Newton's method. Default is False, not using simplified method. See Notes for
            descriptions of what simplified method entails.

        Returns
        -------
        x_n : Tuple[NDArray[np.float64], NDArray[np.int64]]
            Zero point for function estimated from the initial point x_0 and the number of iterations for convergence.
            If Newton's method fails to converge, [np.nan, np.nan] is returned.
        """

        iterations: int = 0

        # Preallocate output array for computing function and jacobian since this requires malloc
        x_n = x_0
        f_n = np.zeros((2,))  # Output array for function values
        f_h = np.zeros((2,))  # Output array for finite differences
        jacobian_n = np.zeros((2, 2))

        # Simplified algorithm creates the inverted jacobian once for the initial guess and uses this for all iterations
        if simplified and jacobian is not None:

            # Compute the inverted jacobian
            jacobian(x_0, jacobian_n)

            # Make sure that the jacobian is not singular
            if np.linalg.det(jacobian_n) != 0:

                # Calculate the inverse jacobian
                jacobian_inv = np.linalg.inv(jacobian_n)

                # Preallocate some values
                function(x_n, f_n)

                # Iterate until newton method converges, diverges above upper limit, or exceeds max iterations
                while loop_tolerance < np.sqrt(f_n[0] ** 2 + f_n[1] ** 2) and iterations < n_iter:
                    # Newtons method for partial derivatives
                    x_n = x_n - jacobian_inv @ f_n

                    # Calculate next iteration's f_n value to avoid extra function call
                    function(x_n, f_n)
                    iterations += 1

        # Simplified algorithm without jacobian matrix.
        elif simplified and jacobian is None:

            # Preallocate some values
            function(x_n, f_n)
            h_x1 = np.array((h, 0))
            h_x2 = np.array((0, h))

            # Let's estimate the jacobian matrix
            jacobian_n[:, 0] = (function(x_n + h_x1, f_h) - f_n) / (h)
            jacobian_n[:, 1] = (function(x_n + h_x2, f_h) - f_n) / (h)

            # Make sure that the jacobian is not singular
            if np.linalg.det(jacobian_n) != 0:

                # Calculate the inverse jacobian
                jacobian_inv = np.linalg.inv(jacobian_n)

                # Iterate until newton method converges, diverges above upper limit, or exceeds max iterations
                while loop_tolerance < np.sqrt(f_n[0] ** 2 + f_n[1] ** 2) and iterations < n_iter:
                    # Newtons method for partial derivatives
                    x_n = x_n - jacobian_inv @ f_n

                    # Calculate next iteration's f_n value to avoid extra function call
                    function(x_n, f_n)
                    iterations += 1

        # Normal Newton's algorithm for estimating zeros with analytic jacobian
        elif not simplified and jacobian is not None:

            # Preallocate some values
            function(x_n, f_n)

            # Iterate until newton method converges, diverges above upper limit, or exceeds max iterations
            while loop_tolerance < np.sqrt(f_n[0] ** 2 + f_n[1] ** 2) and iterations < n_iter:

                # Calculate the jacobian and function value only once
                jacobian(x_n, jacobian_n)

                # Make sure that all values are finite and the jacobian is not singular
                if np.linalg.det(jacobian_n) == 0:
                    break

                # Newtons method for partial derivatives
                x_n = x_n - np.linalg.inv(jacobian_n) @ f_n

                # Calculate next iteration's f_n value to avoid extra function call
                function(x_n, f_n)
                iterations += 1

        # Normal Newton's algorithm for estimating zeros with estimated jacobian
        elif not simplified and jacobian is None:

            # Preallocate some values
            function(x_n, f_n)
            h_x1 = np.array((h, 0))
            h_x2 = np.array((0, h))

            # Iterate until newton method converges, diverges above upper limit, or exceeds max iterations
            while loop_tolerance < np.sqrt(f_n[0] ** 2 + f_n[1] ** 2) and iterations < n_iter:

                # Let's estimate the jacobian matrix
                jacobian_n[:, 0] = (function(x_n + h_x1, f_h) - f_n) / (h)
                jacobian_n[:, 1] = (function(x_n + h_x2, f_h) - f_n) / (h)

                # If the jacobian is singular, Newton's method fails to converge
                if np.linalg.det(jacobian_n) == 0:
                    break

                # Newtons method for partial derivatives
                x_n = x_n - np.linalg.inv(jacobian_n) @ f_n

                # Calculate next iteration's f_n value to avoid extra function call
                function(x_n, f_n)
                iterations += 1

        if np.any(np.isnan(x_n)) or not np.linalg.norm(f_n) < comp_tolerance:
            return np.array((np.nan, np.nan)), iterations
        else:
            return x_n, iterations
