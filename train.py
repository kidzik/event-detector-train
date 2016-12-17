import os
import numpy as np
import math
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product

# Settings
fdir = "csv"
files = os.listdir(fdir)
input_dim = 15
output_dim = 1

# Build the model
model = Sequential()
model.add(LSTM(input_shape = (input_dim,) ,input_dim=input_dim, output_dim=15, return_sequences=True))
model.add(LSTM(output_dim = output_dim, return_sequences=True))
model.add(TimeDistributed(Dense(output_dim, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Merge inputs from different files together
inputs = np.zeros((len(files), 200, input_dim))
outputs = np.zeros((len(files), 200, output_dim))

n = 0

for i,filename in enumerate(files):
    # print("Processing %s..." % (filename,))
    R = np.loadtxt("%s/%s" % (fdir, filename), delimiter=',')
    if R.shape[0] < 500:
        continue

    X = R[300:500,0:input_dim]
    Y = R[300:500,input_dim:(input_dim + output_dim)]
    # ones = np.array( ([1] * 400, ) ).T
    # Add an extra class 'no label'
    # Y = np.concatenate((Y, ones ), axis=1)
    # Y[:,2] = Y[:,2] - Y[:,1]
    # Y[:,2] = Y[:,2] - Y[:,0]

    if (sum(Y)[0] < 0.01):
        continue
    
    inputs[n,:,:] = X
    outputs[n,:,:] = Y.astype(int)[:,0:1]
    n = n+1

    if n >= 500:
        break

print("Training")

# Fit on the training set
ntrain = int(math.floor(n * 0.9))
model.fit(inputs[0:ntrain,:,:], outputs[0:ntrain,:,:], nb_epoch=100, batch_size=32, verbose=2)

# Test on the test set
scores = model.evaluate(inputs[ntrain:n,:,:], outputs[ntrain:n,:,:])
print("Accuracy: %.2f%%" % (scores[1]*100))
