# Script to mscoco dataloader
# author: satwik kottur
import torch
import json
from random import sample, shuffle, choice
import pdb
import math
import numpy as np
from tqdm import tqdm as progressbar
from dataloader import AbstractLoader

class Dataloader(AbstractLoader):
    # Initializer
    def __init__(self, options):
        # Read the file
        with open(options['inputData'], 'r') as fileId:
            content = torch.load(fileId);

        # Convert to torch tensor
        self.text = content['text'];
        self.image = content['image'];
        # Get the image feature size
        self.featSize = content['image'].values()[0].size(1);

        self.numImgs = {};
        for dtype in self.image:
            self.numImgs[dtype] = self.image[dtype].size(0);

        # Transfer params and options
        for key, val in options.items(): setattr(self, key, val);
        for key, val in content['params'].items(): setattr(self, key, val);

        # vocab size
        self.vocabSize = len(self.word2ind);
        # feature size for image features
        self.featSize = self.image['train'].size(1);
        print('Vocab Size: %d' % self.vocabSize)

    def getTrainBatch(self):
        # Pick a set of random images
        batchImgIds = sample(range(self.numImgs['train']), self.batchSize);

        # compile batch
        batch = self.compileBatch(batchImgIds, 'train');
        # Add random negative instances
        numPos = batch['pos'].size();
        batch['neg'] = torch.LongTensor(numPos).random_(self.vocabSize - 1);
        if self.useGPU: batch['neg'] = batch['neg'].cuda();

        return batch;

    # Get test batch
    def getTestBatch(self, startId, dtype):
        # Pick the range of images
        endId = min(self.numInst[dtype], startId + self.batchSize);
        batchImgIds = range(startId, endId);

        # get the set length
        if self.setSize > 0: setLen = self.setSize;
        else: setLen = self.evalSize;

        # Check if they have atleast self.evalSize + 1 number of elements
        batchImgIds = [ii for ii in batchImgIds if \
                        len(self.text[dtype][self.imgOrder[dtype][ii]]) \
                                                                > setLen];
        # TODO: change this back
        # This is not correct!
        #batchImgIds = [ii for ii in batchImgIds if \
        #                len(self.text[dtype][self.imgOrder[dtype][ii]]) > 5];
        batchImgs = [self.imgOrder[dtype][ii] for ii in batchImgIds];

        # Compile batch for test
        numPos = len(batchImgIds);
        setInst = torch.LongTensor(numPos, setLen).fill_(0);
        # handle the case of zero set length
        if setLen == 0: setInst = setInst.view(-1, 1);
        imageFeats = torch.FloatTensor(numPos, self.featSize);

        # Set the set and positive instances
        posInst = [];
        for index, (imgFeatId, imgId) in enumerate(zip(batchImgIds, batchImgs)):
            datum = self.text[dtype][imgId];

            posInst.append(datum[setLen:]);
            if setLen > 0: setInst[index, :] = torch.LongTensor(datum[:setLen]);
            imageFeats[index, :] = self.image[dtype][imgFeatId];

        # Adjust according to gpu/cpu
        if self.useGPU:
            batch = {'set': setInst.cuda(), \
                    'image': imageFeats.cuda()};
        else:
            batch = {'set': setInst.contiguous(), \
                    'image': imageFeats.contiguous()};
        # if set length == 0, ignore the set
        batch['imageId'] = batchImgs;
        batch['pos'] = posInst;
        if self.setSize == 0: batch['set'].fill_(0);
        batch['end'] = endId;

        return batch;

    # Compile batch, given the corresponding image indices and dtype
    def compileBatch(self, batchImgIds, dtype):
        batchImgs = [self.imgOrder[dtype][ii] for ii in batchImgIds];

        # Get lengths from various images
        lengths = [len(self.text[dtype][ii]) for ii in batchImgs];
        # get median and inflate all examples
        medianLen = np.max(lengths);

        if self.setSize > 0: setLen = self.setSize;
        # Change the set length based on requirement
        elif medianLen > 0: setLen = np.random.randint(1, medianLen);
        else: setLen = self.evalSize;
        posInstLen = [ii if ii < setLen else ii - setLen \
                                                    for ii in lengths];
        #numPos = np.sum(lengths) - setLen * len(lengths) - ;
        numPos = np.sum(posInstLen);

        setInst = torch.LongTensor(numPos, setLen).fill_(0);
        # take care of zero length
        if setLen == 0: setInst = setInst.view(-1, 1);
        posInst = torch.LongTensor(numPos, 1);
        imageFeats = torch.FloatTensor(numPos, self.featSize);

        # construct the set and positive instances
        count = 0;
        count2 = 0;
        for imgFeatId, imgId in zip(batchImgIds, batchImgs):
            datum = self.text[dtype][imgId];
            shuffle(datum)

            # given instance its own set + pos instance
            if len(datum) < setLen:
                for ii in xrange(len(datum)):
                    posInst[count] = datum[ii];
                    #if setLen > 0:
                        #setSample = np.random.choice(datum, setLen);
                        #setInst[count, :] = torch.LongTensor(setSample);
                    # get image features
                    imageFeats[count, :] = self.image[dtype][imgFeatId];
                    count += 1;

            else:
                for ii in xrange(setLen, len(datum)):
                    posInst[count] = datum[ii];
                    if setLen > 0:
                        setInst[count, :] = torch.LongTensor(datum[:setLen]);
                    # get image features
                    imageFeats[count, :] = self.image[dtype][imgFeatId];
                    count += 1;

        # Adjust according to gpu/cpu
        if self.useGPU:
            batch = {'set': setInst.cuda(), \
                    'pos': posInst.cuda(), \
                    'image': imageFeats.cuda()};
        else:
            batch = {'set': setInst.contiguous(), \
                    'pos': posInst.contiguous(), \
                    'image': imageFeats.contiguous()};

        # if set length == 0, ignore the set
        batch['imageId'] = batchImgs;

        return batch;
