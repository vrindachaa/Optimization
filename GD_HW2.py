import pandas as pd
import numpy as np

df = pd.read_csv('myFileForHW2 (2).txt', header= None)



y = df.iloc[:, 1]
N= len(x)


A= np.zeros((N, 1))
b = np.ones((N, 1))
A = np.column_stack(((x)**2, -(y)**2))


#step size
mu = 1/np.linalg.norm(A.T @A)
#gradient
def gradient(c):
    grad = 2*(A.T) @ ((A @ c) - b)
    return grad

def GD(error):
    # initialization

    a = np.zeros((2, 1))

    #stopping criteria
    while np.linalg.norm(gradient(a)) > error:
        a = a - gradient(a)*mu
    return a

print(GD((10**-6)))