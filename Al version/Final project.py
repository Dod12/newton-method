import fp_aux as fp
import numpy as np
import matplotlib.pyplot as mpl


#%%
# def fcn(x:np.array):
#     x1,x2=x[0], x[1]
#     f1 = x1**3 - 3*x1*x2**2 -1
#     f2 = 3* x2*x1**2 - x2**3
#     return np.array([f1,f2])
# test= fp.Fractal2D(fcn)
# x=np.array([1,0])
# #print(test.Jacobean(x))
# #print(test.NewtonMethod(x))
# #print(fcn(x))
# test.plot(100,(-10,10,-10,10))

#%%
def fca(x:np.array):
    x1,x2=x[0],x[1]
    f1=x1**3- 3*x1*x2**2 - 2*x1 -2
    f2= 3* x1**2 *x2 - x2**3 - 2*x2
    return np.array([f1,f2])
test2=fp.Fractal2D(fca)
test2.plot(200,(-2.,2.,-2.,2.))

#%%
#def fcb(x:np.array):
    #x1,x2=x[0],x[1]
    #f1= x1**8- 28
    #f2= 3* x1**2 *x2 - x2**3 - 2*x2
    #return np.array([f1,f2])
