import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
import timeit as ti
import matplotlib as mp

class Fractal2D:
    def __init__(self, fcn , jacob=None):
        """
        Parameters
        ----------
        fcn : Function
            The function which is to be evaluated. This must give its output in 
            an array, length 2. 

        Returns
        -------
        None.

        """
        self.fcn = fcn
        self.jacob=jacob
        self.zeroes = [np.NaN]
        
    def Jacobean(self, X: np.array, h=1.e-5):
        """
        Parameters
        ----------
        X: array
            the point at which the Jacobean is evaluated

        Returns
        -------
        Jac : array,
            2 by 2 array giving the Jacobean. 

        """
        x1=X[0]
        x2= X[1]
        
        d1= (self.fcn(np.array([x1+h,x2]))-self.fcn(X))/h
        d2= (self.fcn(np.array([x1,x2+h]))-self.fcn(X))/h
        
                   
        Jac= np.array( [[d1[0],d2[0]],
                        [d1[1],d2[1]]])
        return Jac
    
    def NewtonMethod(self, X0: np.array , tol=1.0e-8):
        """
        

        Parameters
        ----------
        X0 : np.array
            Initial value
        tol : float, optional
            Error tolerance of the zero points. The default is 1.0e-12.

        Returns
        -------
        index : int
            returns 0 if it does not converge, elsewhise the number in list. 
            'hello ' is the error message, which is here solely to indicate to ourselves that things have gone wrong. 

        """
        x=X0
        index='hello'
        for s in range(1000):
            jac= self.Jacobean(x)
            if np.linalg.det(jac)==0:
                index=f'zero {s}'
                break
            invjac= np.linalg.inv(jac)
            x_new = x - invjac @ self.fcn(x)
            x=x_new
            if np.abs(self.fcn(x)[0])<tol and np.abs(self.fcn(x)[1])<tol:
                # I need to check if it's in zeroes, within tolerance
                # If not, then append it to list 
                for t in range(1,len(self.zeroes)):
                    z=self.zeroes[t]
                    if np.abs(z[0]-x[0])<0.1 and np.abs(z[1]-x[1])<0.1:
                        index= t
                        break
                else:
                    self.zeroes.append(x)
                    index = len(self.zeroes)-1
                break
            
            if np.abs(self.fcn(x)[0])>1.e+5 or np.abs(self.fcn(x)[1])>1.e+5:
                index = 0
                break
        else:
            index=0
        return index
    
    def plot(self, N:int, ends: tuple):
        """
        

        Parameters
        ----------
        N : int
            Resolution of the plot
        ends : tuple
            endpoints of plot, tuple of four values a,b,c,d

        Returns
        -------
        None.

        """
        a,b,c,d=ends[0],ends[1],ends[2],ends[3]
        x_vals=np.linspace(a,b,num=N)
        y_vals=np.linspace(c,d,num=N)
        X,Y=np.meshgrid(x_vals, y_vals , indexing='ij', sparse=True)
        gah=[[[e,f]for f in y_vals] for e in x_vals]
        p=np.array(gah)
        A=np.array([[self.NewtonMethod(np.array([e,f]))for f in y_vals]for e in x_vals])
        #now I have the matrix, and I thus need to colour it. wish me luck. 
        mpl.figure()
        mpl.pcolor(A)
    
        