from __future__ import division
import numpy as np
import math
from sys import argv
import matplotlib.pyplot as plt

def calc_probability(x,class_no):
    global T, P,c,weights,covars,means,Pr,hit,miss, confusionMatrix
    # x = T[i]
    # print x
    l = [0 for num in range(6)]
    for k in range(6):
        for j in range(c):
            t1 = weights[k][j]
            root_det = math.sqrt(abs(np.linalg.det(covars[k][j])))
            den = (2*math.pi)**(d/2) * root_det
            t2 = 1/den
            diff = x - means[k][j]
            cinv = np.linalg.inv(covars[k][j])
            t3 = (diff.dot(cinv)).dot(diff)
            t3 = math.exp(-1/2*t3)
            l[k] += t1*t2*t3
        l[k] *= Pr[k]

    max = 0
    max_index = -1
    for k in range(6):
        if l[k] > max:
            max = l[k]
            max_index = k
    if max_index == class_no:
        hit[class_no]+=1
    else:
        miss[class_no]+=1

def begin(components):
    global T,c,d,weights,covars,means,Pr,hit,miss
    perc_used = []
    error_perc = []
    for percentage in range(10,110,10):
        weights = []
        means = []
        covars = []
        P = [1407, 1286, 1374, 1226, 986, 1073]
        Pr = [p/7352 for p in P]
        miss = [0 for i in range(6)]
        hit = [0 for i in range(6)]
        for i in range(6):
            X = np.load(components+'_component/class_'+str(i)+'_parameters_'+str(percentage)+'.npz')
            weights.append(X['arr_0'])
            means.append(X['arr_1'])
            covars.append(X['arr_2'])

        T = np.load('test.npy')
        output_labels = np.load('test_labels.npy')

        c = covars[0].shape[0]
        d = covars[0].shape[1]

        for sample_no in xrange(T.shape[0]):
            calc_probability(T[sample_no],int(output_labels[sample_no]))

        print "misses:",miss
        misses = sum(miss)
        print misses
        print "hits:",hit
        hits = sum(hit)
        print hits
        perc_used.append(percentage)
        error_perc.append(100*misses/(misses+hits))
        print error_perc[-1],"%"

        print "\n"

    plt.plot(perc_used,error_perc,label='Validation')
    plt.xlabel('Percentage of train data used for training')
    plt.ylabel('Error rate of classification for training data')
    plt.legend()
    plt.show()
