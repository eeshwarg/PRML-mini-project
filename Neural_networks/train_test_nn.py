import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from keras import regularizers

def populate_data(s):
    x = np.load('Data/'+s+'.npy')
    labels = np.load('Data/'+s+'_labels.npy')
    y = np.zeros((labels.shape[0],6))
    for i in xrange(len(labels)):
        y[i,int(labels[i])] = 1
    return x,y

(x_train,y_train) = populate_data('train')
(x_val,y_val) = populate_data('val')
(x_test,y_test) = populate_data('test')

model = Sequential()
model.add(Dense(10, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1),input_dim=561))
# model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(6, activation='softmax'))

opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=200,
          validation_data=(x_val,y_val),
          batch_size = 128
          )
          
print '\n',model.metrics_names
score_train = model.evaluate(x_train,y_train,batch_size=128)
print score_train
score_val = model.evaluate(x_val,y_val,batch_size=128)
print score_val
score = model.evaluate(x_test, y_test,batch_size=128)
print score
