# class containing dataloader for lda
# author: satwik kottur
import torch
import json
import random
import pdb
import math
import numpy as np
from tqdm import tqdm as progressbar
#import matplotlib.pyplot as plt
from dataloader import AbstractLoader

# Preloader for LDA
class Dataloader(AbstractLoader):
    # Initializer
    def __init__(self, options):
        # Read the file
        with open(options['inputData'], 'r') as fileId:
            content = json.load(fileId);

        # Convert to torch tensor
        self.data = {};
        for dtype, blob in content['data'].iteritems():
            self.data[dtype] = torch.LongTensor(blob);

        # Transfer params and options
        for key, val in options.items(): setattr(self, key, val);
        for key, val in content['params'].items(): setattr(self, key, val);

        # vocab size
        self.vocabSize = len(self.word2ind);
        print('Vocab Size: %d' % self.vocabSize)

    def getTrainBatch(self):
        inds = torch.LongTensor(self.batchSize).random_(self.numInst['train']-1);
        maxLen = self.data['train'].size(1);
        randLen = np.random.randint(1, maxLen);
        #randLen = self.evalSize;
        batch = self.data['train'].index_select(0, inds);

        numRows = batch.size(0);
        numCols = batch.size(1);
        for row in xrange(numRows):
            shuffle = torch.randperm(numCols);
            for col in xrange(numCols):
                batch[row, col] = batch[row, shuffle[col]];
        # Separate set and ground truth
        setInst = batch[:, :randLen];
        #posInd = torch.LongTensor(self.batchSize, 1).random_(randLen, maxLen-1);
        #posInst = batch.gather(1, posInd).squeeze();
        posInst = batch[:, randLen:];
        negInst = torch.LongTensor(posInst.size()).random_(self.vocabSize-1);

        # Adjust according to gpu/cpu
        if self.useGPU:
            return {'set': setInst.cuda(), \
                    'pos': posInst.cuda(), \
                    'neg': negInst.cuda()};
        else:
            return {'set': setInst.contiguous(), \
                    'pos': posInst.contiguous(), \
                    'neg': negInst.contiguous()};

    # Get test batch
    def getTestBatch(self, startId, dtype):
        endId = min(startId + self.batchSize, len(self.data[dtype]));
        batch = self.data[dtype][startId:endId];

        setInst = batch[:, :self.evalSize];
        posInst = batch[:, self.evalSize:];

        # Adjust according to gpu/cpu
        if self.useGPU:
            return {'set': setInst.cuda(), \
                    'pos': posInst.cuda(),\
                    'end': endId};
        else:
            return {'set': setInst.contiguous(), \
                    'pos': posInst.contiguous(), \
                    'end': endId};

