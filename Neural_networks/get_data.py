import numpy as np

def get_data():
    data = np.genfromtxt('../Data/train_trunc.csv',dtype=float, delimiter = ',')
    test_data = np.genfromtxt('../Data/test_trunc.csv',dtype=float, delimiter = ',')
    np.random.shuffle(data)
    n = data.shape[0]
    train = data[:int(n*0.8),:]
    val = data[int(n*0.8):,:]

    train_labels = train[:,-1]
    val_labels = val[:,-1]
    train = train[:,:-1]
    val = val[:,:-1]
    test_labels = test_data[:,-1]
    test = test_data[:,:-1]

    np.save('Data/train.npy',train)
    np.save('Data/train_labels.npy',train_labels)
    np.save('Data/val.npy',val)
    np.save('Data/val_labels.npy',val_labels)
    np.save('Data/test.npy',test)
    np.save('Data/test_labels.npy',test_labels)

get_data()
