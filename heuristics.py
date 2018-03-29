import numpy as np

def velocity_pred(inputs, nseqlen, output_dim = 0):
    outputs = np.zeros((inputs.shape[0], inputs.shape[1], 1))

    for i in range(inputs.shape[0]):
        try:
            v = inputs[i][:,63] #+9
            v = np.sign(v)
            if output_dim == 0:
                k = np.where(-2 == v[1:] - v[:-1])
            if output_dim == 1:
                k = np.where(2 == v[1:] - v[:-1])
            k = k[0]
            for j in k:
                outputs[i,j,0] = 1
            # k = peakdet(v,0.5)[0][1][0]
        except:
            pass
        
    return outputs

def coordinate_pred(inputs, nseqlen, output_dim = 0):
    outputs = np.zeros((inputs.shape[0], inputs.shape[1], 1))

    for i in range(inputs.shape[0]):
        try:
            if output_dim == 0:
                peaks = peakdet(inputs[i][:,42],0.5)[0]
                index, value = max(enumerate(peaks), key=lambda x: x[1][1])
            if output_dim == 1:
                peaks = peakdet(inputs[i][:,42],0.5)[1]
                index, value = max(enumerate(peaks), key=lambda x: x[1][1])
            
            
            outputs[i,int(peaks[index][0]),0] = 1
        except:
            pass
        
    return outputs

