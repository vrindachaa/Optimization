import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from time import time

#prepping the data, splitting it into two sections (labels&pixels)
df_01 = pd.read_csv('df_49.csv', header=None)
df_01 = df_01.drop(df_01.columns[0], axis= 1)
df_01 = df_01.drop(df_01.index[0], axis= 0)
labels = df_01.iloc[:,0]
pixels = df_01.drop(df_01.columns[0], axis = 1)

#rewriting the labels to represent 1 if 9 and -1 if 4
labels = (-1 if (x == 4)  else 1 for x in labels)
labels = np.asarray(pd.DataFrame(labels, dtype= float))
b = 0.0001

# step size
mu = 10**-4
#pixels_49 = []


def gradient(w, X, y):
    N = X.shape[0]
    summands= (1/N)*np.array([-c*x/(1+np.exp(w.T @ (x*c))) for (x,c) in zip(X.iloc,y)])
    # Sum the vectors, not all of the entries of the matrix
    return np.sum(summands, axis=0)


def function(w, X,y):
    N = X.shape[0]
    summands = (1/N)*np.array([np.log(1+np.exp(-w.T @ (x*c))) for (x,c) in zip(X.iloc,y)])
    return (np.sum(summands))

def GD(n_steps, a, tol, method):
    n = 0

    # stopping criteria
    f = list()
    g = list()
    while n < n_steps:
        current_grad = gradient(a, pixels, labels)
        if norm(current_grad) <= tol:
            break

        if method == 'gradient':
            p= current_grad
            a = (a.T + (mu * -p)).T

        if method == 'momentum':
            p = current_grad
            if n ==0:
                a_prev = a
                a = (a.T + (mu* -p)).T
            else:
                a_new_prev = a
                a = (a.T + (mu* -p)).T + b*(a-a_prev)
                a_prev = a_new_prev




        f.append(function(a,pixels,labels))
        n += 1
        # plot the function
    return a,f

# initialization
v= np.zeros((784,1))
j1,GD_loss = (GD(16,v, tol=10**(-8),method='momentum'))

fig, ax= plt.subplots(1,1, figsize=(10,10))
ax.plot(GD_loss,'-o')
ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel("Loss", fontsize=20)
ax.set_title('Gradient Descent with Momentum Loss Curve, B= ' +str(b) + ', Mu= ' + str(mu), fontsize=22)
plt.show()
j1= pd.DataFrame(j1)
j1.to_csv('weights011.csv')
