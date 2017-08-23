from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt

def sigm(x):
    try:
        ans = math.exp(-x)
    except OverflowError:
        print "Overflow"
        ans = float("inf")
    x = 1/(1+ans)

    return x

def get_accuracy(X,X_labels,theta):
    x_plot = []
    y_plot = []
    for perc in xrange(10):
        hits = 0
        misses = 0
        for i in xrange(X.shape[0]):
            min = 1
            # print "actual class =",X_labels[i]
            for class_no in xrange(6):
                s = sigm(theta[class_no][perc].dot(X[i]))
                # print s
                if s < min:
                    min = s
                    min_index = class_no
                # print min_index
            if min_index == X_labels[i]:
                hits += 1
            else:
                misses += 1

        error_rate = misses/(hits+misses)
        print "Accuracy =",hits/(hits+misses)
        print "Error percentage =",error_rate
        x_plot.append((perc+1)*10)
        y_plot.append(error_rate*100)

    return (x_plot,y_plot)

def get_data():
    train = np.load('Data/train_lda.npy')
    train_labels = np.load('Data/train_labels.npy')
    val = np.load('Data/val_lda.npy')
    val_labels = np.load('Data/val_labels.npy')
    test = np.load('Data/test_lda.npy')
    test_labels = np.load('Data/test_labels.npy')

    train = np.concatenate((np.ones((train.shape[0],1)),train),1)
    val = np.concatenate((np.ones((val.shape[0],1)),val),1)
    test = np.concatenate((np.ones((test.shape[0],1)),test),1)

    theta = []
    for i in xrange(6):
        theta.append(np.load('theta_'+str(i)+'.npy'))

    (xPlot,yPlot) = get_accuracy(train,train_labels,theta)
    (xPlotVal,yPlotVal) = get_accuracy(val,val_labels,theta)
    # (xPlotTest,yPlotTest) = get_accuracy(test,test_labels,theta)

    plt.xlabel('Percentage of train data')
    plt.ylabel('Error percentage')
    plt.plot(xPlot,yPlot,'r',label='Train')
    plt.plot(xPlotVal,yPlotVal,'b',label='Validation')
    # plt.plot(xPlotTest,yPlotTest,'g',label='Test')
    plt.legend()
    plt.show()

get_data()
