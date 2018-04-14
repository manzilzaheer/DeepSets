# Script to test a saved model for deep sets
# author: satwik kottur
from gensim.models import word2vec
import torch
import torch.nn as nn
from options import readOptions
from model import Deepset
from utilities import saveTopWords, evaluateTagging, savePredictedTags
import pdb
import pickle

# read options
allOptions = readOptions();
# Only retain few flags
retainFlags = ['inputData', 'resultPath', 'loadPath', 'setSize',\
                'useGPU', 'batchSize', 'evalSize', 'dataset'];
options = {};
for flag in retainFlags: options[flag] = allOptions[flag];

# Import dataloader based on the options
if options['dataset'] == 'lda':
    from loader.lda import Dataloader;
elif options['dataset'] == 'coco':
    from loader.coco import Dataloader;

# Initialize dataloader
dl = Dataloader(options);

# Setup model
model = Deepset(options['loadPath']);

print('\nArchitecture:')
print(model)

# Ship to gpu if needed
if options['useGPU']: model = model.cuda();

# Evaluation on test
metrics, scores, groundTruth = model.evaluate(dl, 'test');
#evaluateTagging(scores, groundTruth['gtLabels']);
#savePath = '%s-%s.pickle' % (options['loadPath'].replace('/', '-'),
#                            options['evalSize']);
#with open(savePath, 'w') as fileId:
#    pickle.dump({'scores':scores, 'gt':groundTruth}, fileId);

#loadPath = 'models-models-23-Feb-2017-21:39:00-best_model.t7-5.pickle';
#with open(loadPath, 'r') as fileId:
#    data = pickle.load(fileId);
#savePredictedTags(data['scores'], data['gt'], dl);
#savePredictedTags(scores, groundTruth, dl);

'''
# save as mat file
import scipy.io as sio
#sio.savemat('scores_pytorch.mat', {'scores':scores});
import h5py
import numpy as np
from scipy.stats import rankdata
from utilities import computeRankStats

fileId = h5py.File('ls-scores.mat', 'r');
scores = np.array(fileId['PREDs']);
ranks = np.apply_along_axis(rankdata, 1, -1*scores);
# pull out GT ranks
gtRanks = [];
for index, imgInd in enumerate(dl.imgOrder['test']):
    datum = dl.text['test'][imgInd];
    gtRanks.extend([ranks[index, ii] for ii in datum]);
# compute rank statistics
metrics = computeRankStats(np.array(gtRanks));
'''

'''
# save as mat file
import h5py
import numpy as np
from scipy.stats import rankdata
from utilities import computeRankStats

gtRanks = [];
for index, imgInd in enumerate(dl.imgOrder['test']):
    datum = dl.text['test'][imgInd];
    gtRanks.extend(xrange(1, len(datum)+1));
# compute rank statistics
metrics = computeRankStats(np.array(gtRanks));
'''
# Save top words to result file
#saveTopWords(results, dl, 'val');
