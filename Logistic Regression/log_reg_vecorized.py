from __future__ import division
import math
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

one_class = int(sys.argv[1])

alpha = 0.0001
EPSILON = 10**-10
lmbda = 1
#
# theta = np.random.rand(d)
#
# labels = np.array([0,0,1,1])

X = None; labels = None

def initialize(perc):
    global X, labels

    train = np.load('Data/train_lda.npy')
    train_labels = np.load('Data/train_labels.npy')
    val = np.load('Data/val_lda.npy')
    val_labels = np.load('Data/val_labels.npy')
    test = np.load('Data/test_lda.npy')
    test_labels = np.load('Data/test_labels.npy')

    # print perc, type(perc)
    train = train[:int(train.shape[0]*perc/100),:]
    train_labels = train_labels[:int(train_labels.shape[0]*perc/100)]


    train = np.concatenate((np.ones((train.shape[0],1)),train),1)
    val = np.concatenate((np.ones((val.shape[0],1)),val),1)
    test = np.concatenate((np.ones((test.shape[0],1)),test),1)

    X = train[train_labels == one_class]
    labels = np.zeros(np.count_nonzero(train_labels == one_class))
    # labels = train_labels[train_labels == 2]
    for i in [j for j in range(6) if j!=one_class]:
        X = np.concatenate((X,train[train_labels == i]),0)
        count = np.count_nonzero(train_labels == i)
        # print count
        o = np.ones(count)
        labels = np.concatenate((labels,o),0)
        # print X.shape
        # print labels.shape
    theta = np.random.rand(train.shape[1])
    # print theta

    return theta

def get_data():
    data = np.genfromtxt('../Data/train_trunc.csv',dtype=float, delimiter = ',')
    # print type(data)
    # print data.shape
    np.random.shuffle(data)
    n = data.shape[0]
    d = data.shape[1]
    train = data[:int(n*0.8),:]
    val = data[int(n*0.8):,:]
    # print train.shape
    # print val.shape

    train_labels = train[:,-1]
    val_labels = val[:,-1]
    train = train[:,:-1]
    val = val[:,:-1]

    return (train, train_labels, val, val_labels)

def sigmoid(arr):
    for element in arr:
        try:
            ans = math.exp(-element)
        except OverflowError:
            print "Overflow"
            ans = float("inf")
        element = 1/(1+ans)

            # exit(1)
    return arr
    # for element in arr:
    #     element = 1/(1+math.exp(-element))
    # return arr
    # return 1/(1+np.exp(-arr))

def sigm(x):
    try:
        ans = math.exp(-x)
    except OverflowError:
        print "Overflow"
        ans = float("inf")
    x = 1/(1+ans)

    return x

def gradient_descent(perc):
    theta = initialize(perc)
    global X, labels, lmbda
    count = 0
    while True:
        # print theta
        count+=1
        h = sigmoid(X.dot(theta))
        e = h - labels
        reg = np.copy(theta)
        reg[0] = 0
        der = X.T.dot(e) + lmbda*reg
        temp_theta = theta - alpha*der
        # print np.sum(np.absolute(der))
        if np.sum(np.absolute(der)) < EPSILON:
            break
        theta = temp_theta
    # print theta
    # print count
    np.save('theta.npy',theta)

    return theta

def test_on_val(theta):
    hits = 0; misses = 0
    val = np.load('Data/val_lda.npy')
    val_labels = np.load('Data/val_labels.npy')
    val = np.concatenate((np.ones((val.shape[0],1)),val),1)
    X = val[val_labels == one_class]
    labels = np.zeros(np.count_nonzero(val_labels == one_class))
    # labels = train_labels[train_labels == 2]
    for i in [j for j in range(6) if j!=one_class]:
        X = np.concatenate((X,val[val_labels == i]),0)
        count = np.count_nonzero(val_labels == i)
        # print count
        o = np.ones(count)
        labels = np.concatenate((labels,o),0)

    for i in range(len(X)):
        det = sigm(theta.dot(X[i]))
        # print "Sigmoid = ",det
        # print "Label =",labels[i]
        if det > 0.5:
            hits += labels[i]
            misses += 1 - labels[i]
        else:
            hits += 1 - labels[i]
            misses += labels[i]
    # print "Hits =",hits
    # print "Misses =",misses
    error_rate = 100*misses/(misses+hits)
    # print "Validation error percentage: ",error_rate

    return error_rate

def test_on_train(theta):
    hits = 0; misses = 0
    global X, labels
    for i in range(len(X)):
        det = sigm(theta.dot(X[i]))
        # print "Sigmoid = ",det
        # print "Label =",labels[i]
        if det > 0.5:
            hits += labels[i]
            misses += 1 - labels[i]
        else:
            hits += 1 - labels[i]
            misses += labels[i]
    # print "Hits =",hits
    # print "Misses =",misses
    error_rate = 100*misses/(misses+hits)
    # print "Train error percentage: ",error_rate
    return error_rate


xPlot = []
yPlot = []
valPlot = []

theta_final = np.empty((10,6))

for perc in xrange(10,110,10):
    theta = gradient_descent(perc)
    # print theta.shape
    theta_final[int(perc/10-1)] = theta
    error_rate = test_on_train(theta)
    val_error = test_on_val(theta)

    xPlot.append(perc)
    yPlot.append(error_rate)
    valPlot.append(val_error)
np.save('theta_'+str(one_class)+'.npy',theta_final)
