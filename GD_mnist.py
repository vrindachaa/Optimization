import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

#prepping the data, splitting it into two sections (labels&pixels)
df_01 = pd.read_csv('df_49.csv', header=None)
df_01 = df_01.drop(df_01.columns[0], axis= 1)
df_01 = df_01.drop(df_01.index[0], axis= 0)
labels = df_01.iloc[:,0]
pixels = df_01.drop(df_01.columns[0], axis = 1)

pixels = pixels.div(255)


#rewriting the labels to represent 1 if 9 and -1 if 4
labels = (-1 if (x == 4)  else 1 for x in labels)
labels = np.asarray(pd.DataFrame(labels, dtype= float))

#beta
b = 0.9

# step size
mu = 10**-5

#gradient tool; takes parameters w = weight vector, X = data, y = labels and returns the gradient of the function
def gradient(w, X, y):
    N = X.shape[0]
    summands= (1/N)*np.array([-c*x/(1+np.exp(w.T @ (x*c))) for (x,c) in zip(X.iloc,y)])
    # Sum the vectors, not all of the entries of the matrix
    return np.sum(summands, axis=0)

#calculates the function value or the loss function ; takes parameters w = weight vector, X = data, y = labels and
# returns the function value at the weight vector
def function(w, X,y):
    N = X.shape[0]

    summands = (1/N)*np.array([np.log(1+np.exp(-w.T @ (x*c))) for (x,c) in zip(X.iloc,y)])
    return (np.sum(summands))

#calculates gradient descent with various methods including momentum and nesterovs, with tol being a tolerance limit for
# convergence, a as the x_not vector to start the iterative process and n_steps for the number of steps;
# returns a weight vector and the a list of the function value at each iterate
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

        if method == 'nesterov':
            if n == 0:
                a_prev = a
                a = (a.T + (mu* -current_grad)).T

            else:
                a_new_prev = a
                a = (a.T + b*(a-a_prev).T - mu*gradient(a + b*(a- a_prev), pixels,labels).T).T
                a_prev = a_new_prev

        if method == 'F-R':
            alpha = 1
            alpha_factor = 0.06
            if n == 0:
                p = -current_grad
            while alpha > 0:

                cond1_RHS= round(function(np.subtract(a.T, alpha*p).T, pixels, labels), 6)
                cond1_LHS = round(function(a, pixels, labels) + 0.8*alpha*np.dot(current_grad, p), 6)
                if (cond1_RHS <= cond1_LHS):

                    cond2_RHS = round((abs(np.dot(gradient(np.subtract(a.T, alpha*p).T, pixels, labels), p))), 6)
                    cond2_LHS = -0.9 * round(np.dot(current_grad, p),6)

                    if (cond2_RHS <= cond2_LHS):
                        alpha = alpha
                        break
                        print('alpha has been found!')
                    else:
                        alpha = alpha_factor*alpha
                        print('Alpha:' + str(alpha))
                        print('Refactoring..')
                        print(str(cond2_RHS) + '>' + str(cond2_LHS))



                else:
                    alpha = alpha_factor*alpha
                    print('Alpha:' + str(alpha))
                    print('Refactoring...')
                    print(str(cond1_RHS) + '>' + str(cond1_LHS))


            a = a + alpha*(-p)
            new_grad = gradient(a, pixels, labels)
            beta = np.dot(new_grad, new_grad)/ np.dot(current_grad,current_grad)
            p = new_grad +beta*p




        f.append((function(a,pixels,labels)))
        n += 1
        # plot the function
    return a,f

# initialization
v= np.zeros((784,1))
j1,GD_loss = (GD(100,v, tol=10**(-8),method='F-R'))


fig, ax= plt.subplots(1,1, figsize=(10,10))
ax.plot(GD_loss,'-o')

ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel("Loss", fontsize=20)
ax.set_title('F-R Loss Curve', fontsize=22)
#B= ' +str(b) + ', Mu= ' + str(mu)
plt.show()
j1= pd.DataFrame(j1)
j1.to_csv('weights011.csv')
