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
            compile: bool = True
    ) -> None:
        """


        Parameters
        ----------
        function : Callable[[NDArray[Real]], NDArray[Real]]
            DESCRIPTION.
        jacobian : Callable[[NDArray[Real]], NDArray[Real]], optional
            DESCRIPTION. The default is None.
        compile : bool, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
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
            loop_tolerance: float = np.finfo(np.float64).eps, comparison_tolerance: float = 1e-9,
            upper_lim: float = 1.e+8, simplified: bool = False) -> NDArray[np.float64]:
        """
        Gives us the zeroes, to be put in the newton_index.

        Parameters
        ----------
        x_0 : NDArray[Real]
            Initial Guess.
        n_iter : int, optional
            Maximum number of iterations. The default is 1000.
        h : float, optional
            If no jacobian is given, we call the _newton_helper_estimate_jacobian and let this be our h.
            The default is 1e-5.
        loop_tolerance : float, optional
            The tolerance of the zero points.
            The default is np.finfo(np.float64).eps.
        comparison_tolerance : float, optional
            If the norm of the function at
            The default is 1.e-9.
        upper_lim : float, optional
            Upper limit of function value for simplified newton method
            The default is 1.e+8
        simplified: bool, optional
            Determines whether we use the simplified newton method
            The default is False

        Returns
        -------
        NDArray[Real]
            The zero points the initial guess converges to, if it diverges it gives nan and nan.

        """
        if self.jacobian is not None:
            x_n = self._newton_helper(self.function, self.jacobian, x_0, n_iter, loop_tolerance, upper_lim, simplified)

        else:
            x_n = self._newton_helper_estimate_jacobian(self.function, x_0, n_iter, h, loop_tolerance, upper_lim,
                                                        simplified)

        if np.any(np.isnan(x_n)) or not np.linalg.norm(self.function(x_n)) < comparison_tolerance:
            warning(RuntimeWarning(f"Newton's method on {x_0} failed to converge in {n_iter} iterations."))
            return np.array((np.nan, np.nan))
        else:
            info(f"Estimated root of {x_0} to {x_n}.")
            return x_n

    def newton_index(self, x0: NDArray[Real], tol: float = 1e-9, up: float = 1.e+8, simple: bool = False,
                     **kwargs) -> int:
        """
        This makes an index of the zeroes we use when making the fractal

        Parameters
        ----------
        x0 : NDArray[Real]
            Initial guess.
        tol : float, optional
            The tolerance of the zerovalue. The default is 1e-9.
        up : float, optional
            if simplified newton method is used, this is upper limit
            The default is 1.e+8
        simple: bool, optional
            if simplified newton method is wanted, set this to True
            The default is False.
        **kwargs : TYPE
            If we've got some jacobians or similar.

        Returns
        -------
        int
            The index of the zero in the list.

        """
        root = self.newton_zeros(x0, **kwargs, comparison_tolerance=tol, upper_lim=up, simplified=simple)
        for i, item in enumerate(self.zeroes):
            if np.linalg.norm(item - root) < tol or (np.any(np.isnan(item)) and np.any(np.isnan(root))):
                return i
        self.zeroes.append(root)
        return len(self.zeroes) - 1

    def plot(self, N: int, a: float, b: float, c: float, d: float, simplified: bool = False, upper_lim=1.e+8, n_cpus=1):
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
        Simplified: Bool
            If we wish to use the simplified newton method, set this as True.
            The default is False.
        upper_lim: Float
            If simplified newton method is used, this is upper limit before it breaks away.
        n_cpus : TYPE, optional
            DESCRIPTION. (What is this Daniel?)
            The default is 1.

        Returns
        -------
        None.

        """

        grid = np.array(np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N)))
        self.A = np.zeros_like(grid[0, ...])
        for i, j in tqdm(itertools.product(range(grid.shape[1]), range(grid.shape[2])), total=N ** 2):
            self.A[i, j] = self.newton_index(grid[:, i, j], simple=simplified, up=upper_lim)
        self.mesh = plt.imshow(self.A, extent=[a, b, c, d], origin="lower", aspect="equal", interpolation=None)
        plt.show(block=True)

        # ToDo: add dependance on Simplified, make some fcns to make a simplified newton method.

    @staticmethod
    @numba.jit(nopython=True)
    def _newton_helper(function: Callable, jacobian: Callable, x_0: NDArray[Real], n_iter: int = 10000,
                       loop_tolerance: float = np.finfo(np.float64).eps, upper_lim: float = 1.e+8,
                       simplified: bool = False) -> NDArray[np.float64]:
        """
        This is used when we have a jacobian given to us.

        Parameters
        ----------
        function : Callable
            the function we wish to find the zero of
        jacobian : Callable
            the jacobian, which we were given
        x_0 : NDArray[Real]
            initial guess
        n_iter : int, optional
            Maximum number of iterations. The default is 10000.
        loop_tolerance : float, optional
            Tolerance of our zeropoints. The default is np.finfo(np.float64).eps.
        upper_lim : float, optional
            upper limit the fcn can grow to before we break off. The default is 1.e+8
        simplified : bool, optional
            Determines whether or not we use the simplified newto method
            The default is False

        Returns
        -------
        x_n : TYPE
            DESCRIPTION.

        """
        iterations = 0
        x_n = x_0

        if not simplified:
            while not np.linalg.norm(function(x_n)) < loop_tolerance and iterations < n_iter:
                if not (np.any(np.isfinite(x_n)) and np.any(np.isfinite(jacobian(x_n))) and np.any(
                        np.isfinite(function(x_n)))) or np.linalg.det(jacobian(x_n)) == 0:
                    break
                x_n = x_n - np.linalg.inv(jacobian(x_n)) @ function(x_n)
                iterations += 1
        else:
            jac = jacobian(x_0)
            while not np.linalg.norm(function(x_n)) < loop_tolerance and iterations < n_iter:
                if np.linalg.norm(function(x_n)) > upper_lim or np.linalg.det(jac) == 0:
                    break
                x_n = x_n - np.linalg.inv(jac) @ function(x_n)
                iterations += 1

        return x_n

    @staticmethod
    @numba.jit(nopython=True)
    def _newton_helper_estimate_jacobian(function: Callable, x_0: NDArray[Real], n_iter: int = 10000,
                                         h: float = 1e-5, loop_tolerance: float = np.finfo(np.float64).eps,
                                         upper_lim: float = 1.e+8, simplified: bool = False) -> NDArray[np.float64]:
        """
        This is for the numerical method of calculating the Jacobian- used if no jacobian is given.

        Parameters
        ----------
        function : Callable
            The function which zero we wish to find
        x_0 : NDArray[Real]
            Initial guess
        n_iter : int, optional
            Maximum number of iterations.
            The default is 10000.
        h : float, optional
            The increment in the numerical estimate of derivative.
            The default is 1e-5.
        loop_tolerance : float, optional
            The tolerance of our zeropoint.
            The default is np.finfo(np.float64).eps.  (question, should we change this?)
        upper_lim : float, optional
            The upper limit of the function value using the simplified newton method
            The default is 1.e+8
        simplified : bool, optional
            Determines whether we use the simplified method.
            The default is False

        Returns
        -------
        x_n : NDArray[Real]
            Our zero point.

        """
        iterations = 0
        x_n = x_0
        if not simplified:
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
        else:
            # estimating the jacobian
            df1_dx1, df2_dx1 = (function(x_0 + np.array([h, 0])) - function(x_0 - np.array([h, 0]))) / (2 * h)
            df1_dx2, df2_dx2 = (function(x_0 + np.array([0, h])) - function(x_0 - np.array([0, h]))) / (2 * h)
            jacob = np.array([[df1_dx1, df1_dx2],
                              [df2_dx1, df2_dx2]])
            if not np.any(np.isfinite(jacob)) or np.linalg.det(jacob) == 0:
                return x_n
            else:
                while not np.linalg.norm(function(x_n)) < loop_tolerance and iterations < n_iter:
                    if np.linalg.norm(function(x_n)) > upper_lim:
                        break
                    x_n = x_n - np.linalg.inv(jacob) @ function(x_n)
                    iterations += 1
            return x_n
