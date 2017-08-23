from __future__ import division
import numpy as np
import math
import time
from sys import argv

c = int(argv[1])
D = n = d = w = mu = sigma = None

# c is the number of Gaussian distributions in the model
def initialize(class_no,train_perc):
    global D,n,d,w,mu,sigma,c
    Data = np.load('train.npz')
    D = Data['arr_'+str(class_no)]
    n = D.shape[0]                                                        # n = number of training samples
    d = D.shape[1]                                                        # d = the dimension of the input data
    D = D[:int(round(n*train_perc/100)),:]
    n = D.shape[0]
    d = D.shape[1]
    print "Percentage:",train_perc
    print "No of training samples: ",D.shape[0]
    w = [1/c for i in range(c)]                                           # Initialize weights equally for all components
    mu = D[np.random.choice(D.shape[0], size=c, replace=False), : ]       # Randomly initializing the means of the Gaussians to c random datapoints
    I = np.identity(d)
    sigma = np.empty((c,d,d))
    for i in range(c):                                                    #Initialize the the covariance matrices for all components to be the identity matrix
        sigma[i] = I

def calc_likelihood(i):
    global D,n,d,w,mu,sigma,c
    L = np.empty(c)
    x = D[i]
    least = float("inf")
    for k in range(c):
        t1 = w[k]
        t2 = 1/((2*math.pi)**(d/2) * math.sqrt(abs(np.linalg.det(sigma[k]))))
        t3 = ((x-mu[k]).dot(np.linalg.inv(sigma[k]))).dot((x-mu[k]))
        t3 = math.exp(-1/2*t3)

        L[k] = t1*t2*t3
    return L


def expectation():
    global D,n,d,w,mu,sigma,c
    P = np.empty((c,n))

    for i in range(n):
        L = calc_likelihood(i)
        den = np.sum(L)
        for k in range(c):
            num = L[k]
            P[k,i] = num/den

    maximization(P)


def maximization(P):
    global D,n,d,w,mu,sigma,c
    for k in range(c):
        w[k]= (np.sum(P[k]))/n

    for k in range(c):
        num = np.zeros(d)
        for i in range(n):
            num += P[k,i]*D[i]
        den = w[k]*n
        mu[k]=num/den

    for k in range(c):
        num = np.zeros((d,d))
        for i in range(n):
            t = D[i]-mu[k]
            num += P[k,i] * (np.outer(t,t))
        den = w[k]*n
        sigma[k] = num/den

for j in range(6):
    print "\n   CLASS",j,"\n"
    for train_perc in range(10,110,10):
        initialize(j,train_perc)
        for i in range(50):
            print "-----------Iteration",i+1,"--------"
            expectation()
            print "Weights: ",w
        print "=======================\n\n"
        print "Means: ",mu,"\n-----------------------\n"

        print "Covariances: ", sigma

        np.savez(str(c)+'_component/class_'+str(j)+'_parameters_'+str(train_perc)+'.npz',w,mu,sigma)
