import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_data():
    data = np.genfromtxt('../Data/train_trunc.csv',dtype=float, delimiter = ',')
    test_data = np.genfromtxt('../Data/test_trunc.csv',dtype=float, delimiter = ',')
    print type(data)
    print data.shape
    np.random.shuffle(data)
    n = data.shape[0]
    d = data.shape[1]
    train = data[:int(n*0.8),:]
    val = data[int(n*0.8):,:]
    print train.shape
    print val.shape

    train_labels = train[:,-1]
    val_labels = val[:,-1]
    train = train[:,:-1]
    val = val[:,:-1]
    test_labels = test_data[:,-1]
    test = test_data[:,:-1]


    lda = LinearDiscriminantAnalysis()
    model = lda.fit(train,train_labels)
    train_lda = model.transform(train)
    val_lda = model.transform(val)
    test_lda = model.transform(test)

    for d in xrange(train_lda.shape[1]):
        s = train_lda[:,d].std()
        train_lda[:,d] /= s

    for d in xrange(val_lda.shape[1]):
        s = val_lda[:,d].std()
        val_lda[:,d] /= s

    for d in xrange(test_lda.shape[1]):
        s = test_lda[:,d].std()
        test_lda[:,d] /= s

    print train_lda.shape
    print train_labels.shape
    print val_lda.shape
    print val_labels.shape
    print test_lda.shape
    print test_labels.shape

    np.save('Data/train_lda.npy',train_lda)
    np.save('Data/train_labels.npy',train_labels)
    np.save('Data/val_lda.npy',val_lda)
    np.save('Data/val_labels.npy',val_labels)
    np.save('Data/test_lda.npy',test_lda)
    np.save('Data/test_labels.npy',test_labels)

    return (train, train_labels, val, val_labels)

get_data()
