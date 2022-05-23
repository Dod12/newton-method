from FinalProject import Fractal2D, Fractal2D_old
import numpy as np
from logging import getLogger
import matplotlib.pyplot as plt

getLogger("root").setLevel("WARNING")


def function1(X: np.ndarray) -> np.ndarray:
    return np.array([X[0] ** 3 - 3 * X[0] * X[1] ** 2 - 1,
                     3 * (X[0] ** 2) * X[1] - X[1] ** 3])


def jacobian1(X: np.ndarray) -> np.ndarray:
    return np.array([[3 * X[0] ** 2 - 3 * X[1] ** 2, -6 * X[0] * X[1]],
                     [6 * X[0] * X[1], 3 * X[0] ** 2 - 3 * X[1] ** 2]])


fractal1 = Fractal2D(function=function1, jacobian=jacobian1, compile=True)
fractal1.plot(100, -1, 1, -1, 1)


# fractal2.plot(100, -1, 1, -1, 1)

def fcn_3(X: np.ndarray) -> np.ndarray:
    return np.array([
        (X[0] ** 8 - 28 * X[0] ** 6 * X[1] ** 2 + 70 * X[0] ** 4 * X[1] ** 4 + 15 * X[0] ** 4 - 28 * X[0] ** 2 * X[1]
         ** 6 - 90 * X[0] ** 2 * X[1] ** 2 + 15 * X[1] ** 4 - 16),
        (8 * X[0] ** 7 * X[1] - 56 * X[0] ** 5 * X[1] ** 3 + 56 * X[0] ** 3 * X[1] ** 5 + 60 * X[0] ** 3 * X[1] - 8 *
         X[0] * X[1] ** 7 - 60 * X[0] * X[1] ** 3)
    ])


fractal3 = Fractal2D(function=fcn_3, compile=True)
# fractal4 = Fractal2D_old(function=fcn_3)
# print("Second equation")
# fractal3.plot(100,-1,1,-1,1)
# fractal4.plot(100,-1,1,-1,1)
