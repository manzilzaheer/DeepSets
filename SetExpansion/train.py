# Script to train deep sets
# author: satwik kottur
from gensim.models import word2vec
import torch
from options import readOptions
import random
from model import Deepset
from loader import *
import torch.optim as optim
from time import gmtime, strftime
from utilities import composeEmbeddings, evaluateTagging
import pdb
import pickle
import numpy as np

#####################################################################
# read options
options = readOptions();

# Seed for random, numpy and torch
random.seed(1234);
torch.manual_seed(1234);
np.random.seed(1234);

#####################################################################
# Import dataloader based on the options
if options['dataset'] == 'lda':
    from loader.lda import Dataloader;
elif options['dataset'] == 'coco':
    from loader.coco import Dataloader;

# Initialize dataloader
dl = Dataloader(options);

#####################################################################
# Setup model parameters
modelParams = options;
keys = ['vocabSize'];
for key in keys: modelParams[key] = getattr(dl, key);
if hasattr(dl, 'featSize'): modelParams['featSize'] = dl.featSize;

# Setup model
model = Deepset(modelParams);
# Check for initializations
if modelParams['embedPath'] != None:
    print('Loading word embeddings: %s' % modelParams['embedPath'])
    model.initEmbeddings(composeEmbeddings(dl));
print('\nArchitecture:')
printArgs = ['embedder', 'postEmbedder', 'combine', 'imgTransform', 'pooler'];
for arg in printArgs:
    if hasattr(model, arg):
        print('%s: ' % arg)
        print(getattr(model, arg))

# Ship to gpu if needed
if options['useGPU']: model = model.cuda();

#####################################################################
# Setup optimizer
# layers to freeze
#toFreeze = ['embedder'];
toFreeze = [];
for name in toFreeze:
    module = getattr(model, name);
    for p in module.parameters(): p.requires_grad = False;

# layers to optimize
#toLearn = ['postEmbedder', 'combine', 'imgTransform'];
toLearn = ['embedder', 'postEmbedder', 'combine', 'imgTransform'];
optimArgs = []; # arguments to optimizer
for name in toLearn:
    # if module not found, ignore
    if not hasattr(model, name): continue;
    module = getattr(model, name);
    optimArgs.append({'params': module.parameters(), \
                        'lr': options['learningRate']});

# initialize the optimizer based on modules to learn
optimizer = optim.Adam(optimArgs);

#####################################################################
# Training
options['iterPerEpoch'] = dl.numInst['train']/options['batchSize'];
print('Number of iterations per epoch: %d' % options['iterPerEpoch'])
loss = None;
#bestMRR = model.evaluate(dl, 'val')[0]['mrr'];
bestMRR = 0;

print('\nTraining:')
for loopId in xrange(1, options['iterPerEpoch'] * options['numEpochs']):
    # Reset gradients, perform training step, update
    optimizer.zero_grad();
    curLoss = model.trainStep(dl);
    optimizer.step();

    # Evaluate for every 10 epochs
    if loopId % (options['evalPerEpoch'] * options['iterPerEpoch']) == 0:
        metrics, scores, gtLabels = model.evaluate(dl, 'val');
        evaluateTagging(scores, gtLabels['gtLabels']);
        curMRR = metrics['mrr'];

        # Save only if mrr improves
        if bestMRR < curMRR:
            bestMRR = curMRR;
            # Save model
            path = options['savePath'] + 'best_model.t7';
            model.save(path);

    if loss == None: loss = curLoss;
    else: loss = 0.95 * loss + 0.05 * curLoss;

    # Print info
    if loopId % 100 == 0:
        epoch = float(loopId) / options['iterPerEpoch'];
        time = strftime("%a, %d %b %Y %X", gmtime())
        print('[%s][Iter: %d][Ep: %.2f] Loss: %.4f' % \
                    (time, loopId, epoch, loss))
#####################################################################
