from FinalProject import Fractal2D, Fractal2D_old
import numpy as np
from logging import getLogger

getLogger("root").setLevel("WARNING")

def function1(X: np.ndarray) -> np.ndarray:
    return np.array([X[0]**3 - 3*X[0]*X[1]**2 - 1,
                     3*(X[0]**2)*X[1] - X[1]**3])

def jacobian1(X: np.ndarray) -> np.ndarray:
    return np.array([[3*X[0]**2 - 3*X[1]**2, -6*X[0]*X[1]],
                     [6*X[0]*X[1], 3*X[0]**2 - 3*X[1]**2]])


fractal1 = Fractal2D(function=function1, jacobian=jacobian1, compile=True)
#fractal2 = Fractal2D_old(function=function2, jacobian=jacobian2)
#print("First equation")
fractal1.plot(100, -1, 1, -1, 1)
#fractal2.plot(100, -1, 1, -1, 1)

def fcn_3(X: np.ndarray)-> np.ndarray:
    x1, x2= X[0], X[1]
    f1=x1**8- 28*x1**6 *x2**2 + 70 * x1**4 *x2**4 +15* x1**4 -28*x1**2 * x2**6 - 90*x1**2 *x2**2+ 15 *x2**4 - 16
    f2=8*x1**7 *x2 -56 *x1**5*x2**3 +56*x1**3*x2**5 +60*x1**3*x2 -8*x1*x2**7- 60*x1*x2**3
    return np.array([f1,f2])
#fractal3 = Fractal2D(function=fcn_3)
#fractal4 = Fractal2D_old(function=fcn_3)
#print("Second equation")
#fractal3.plot(100,-1,1,-1,1)
#fractal4.plot(100,-1,1,-1,1)