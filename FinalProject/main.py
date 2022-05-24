from numpy.typing import NDArray

from FinalProject import Fractal2D, Fractal2D_old
import numpy as np
from logging import getLogger
import matplotlib
matplotlib.use("MacOSX")

getLogger("root").setLevel("WARNING")


def function1(X: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    x1 = X[0]
    x2 = X[1]
    out[0] = x1 ** 3 - 3 * x1 * x2 ** 2 - 1
    out[1] = 3 * x1 ** 2 * x2 - x2 ** 3
    return out


def jacobian1(X: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    x1 = X[0]
    x2 = X[1]
    out[0, 0] = 3 * x1 ** 2 - 3 * x2 ** 2
    out[0, 1] = -6 * x1 * x2
    out[1, 0] = 6 * x1 * x2
    out[1, 1] = 3 * x1 ** 2 - 3 * x2 ** 2
    return out


fractal1 = Fractal2D(function=function1, jacobian=jacobian1, compile=True)
fractal1.plot(100, -1, 1, -1, 1)


# fractal2.plot(100, -1, 1, -1, 1)

def fcn_3(X: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    x1 = X[0]
    x2 = X[1]
    out[0] = x1 ** 8 - 28 * x1 ** 6 * x2 ** 2 + 70 * x1 ** 4 * x2 ** 4 + 15 * x1 ** 4 - 28 * x1 ** 2 * x2 ** 6 - 90 * x1 ** 2 * x2 ** 2 + 15 * x2 ** 4 - 16
    out[1] = 8 * x1 ** 7 * x2 - 56 * x1 ** 5 * x2 ** 3 + 56 * x1 ** 3 * x2 ** 5 + 60 * x1 ** 3 * x2 - 8 * x1 * x2 ** 7 - 60 * x1 * x2 ** 3
    return out


fractal3 = Fractal2D(function=fcn_3, compile=True)
fractal3.plot(100, -1, 1, -1, 1)
#fractal4 = Fractal2D_old(function=fcn_3)
# print("Second equation")
#fractal3.plot(100,-1,1,-1,1)
#fractal4.plot(100,-1,1,-1,1)
