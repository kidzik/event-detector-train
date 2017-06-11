
# coding: utf-8

# # Identification of gait events using LSTM networks

# In[1]:

from utils import *
import json
import itertools

# Settings
input_dim = 99 #27*2
output_dim = 2
nseqlen = 128
train_dir = "/media/lukasz/DATA/train"
epochs = 200

# Load data
inputs, outputs, ids = load_data(train_dir, input_dim, output_dim, nseqlen, nsamples = 100000)

cols = range(15) + [60 + i for i in range(13)] + [12 + 81 + i for i in range(6)]

for output_column, nlayers, nhidden in itertools.product(range(2),range(3),[16,32,64]):
    signature = (output_column, nhidden, nlayers)
    
    model = construct_model(hidden = nhidden, lstm_layers = nlayers, input_dim = len(cols), output_dim = 1)

    history = model.fit(inputs[:,:,cols], outputs[:,:,output_column:(output_column+1)], epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
        
    with open('models/model-%d-%d-%d.json' % signature, 'w') as outfile:
        desc = {"cols": cols, "signature": signature, "history": history.history, "epochs": epochs}
        json.dump(desc, outfile)
        print(desc)
        
    model.save("models/model-%d-%d-%d.h5" % signature )
