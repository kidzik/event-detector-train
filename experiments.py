from utils import *

# Settings
input_dim = 27
output_dim = 1
nseqlen = 128
fdir = "csv"

# Load data
inputs, outputs, ids = load_data(fdir, input_dim, output_dim, nseqlen, nsamples = 10000)
n = inputs.shape[0]
ntrain = int(math.floor(n * 0.9))

# Combinations
hiddens = [128]
lstmlays = [3]
products = list(itertools.product(hiddens, lstmlays))

# Run
results = {}

tt = time.strftime('%Y%m%d%H%M%s')

for exam in products:
    model = construct_model(hidden = exam[0], lstm_layers = exam[1])
    history = model.fit(inputs[0:ntrain,:,:], outputs[0:ntrain,:,:], nb_epoch=2000, batch_size=32, verbose=1, validation_split=0.1)
    plot_history(history)

    # Test on the test set
    scores = model.evaluate(inputs[ntrain:n,:,:], outputs[ntrain:n,:,:])
    print("Test set loss: %.4f" % (scores[0]))
    sdist = []
    for ntrial in range(1000):
        likelihood = res[ntrial,:,:]
        true = outputs[ntrain + ntrial,:,:]
        sdist.extend(eval_prediction(likelihood, true, ids[ntrain + ntrial], plot = False))

    plot_stats(sdist)
    
    results[exam] = (history.history, scores[0])
    model.save("model-%s-%d-%d.h5" % (tt,exam[0],exam[1]))
    
pickle.dump( results, open( "res-%s.p" % (tt,), "wb" ) )
