#sum Class containing deepsets model
# author: satwik kottur
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
import zoo
from utilities import *
from tqdm import tqdm as progressbar
from scipy.stats import rankdata

class Deepset(nn.Container):
    def __init__(self, params):
        super(Deepset, self).__init__();

        # if list of parameters is supplied
        if type(params).__name__ == 'dict': self.initialize(params);
        # if path is provided to preload
        else: self.load(params);

        # transfer parameters to self
        for key, val in self.params.items(): setattr(self, key, val);
        # Criterion
        self.criterion = nn.MarginRankingLoss(self.margin);
        # Additional variables
        self.labels = torch.FloatTensor(1).fill_(1);
        if self.useGPU: self.labels = self.labels.cuda();

    def initialize(self, params):
        self.params = params;

        modelParts = zoo.selectModel(params);
        flags = ['imgTransform', 'combine', 'embedder', 'postEmbedder'];
        # refine flags
        for flag in flags:
            if flag not in modelParts: print('Missing: %s'%flag)
            else: setattr(self, flag, modelParts[flag]);

        # define word transform as composition
        self.wordTransform = lambda x: self.postEmbedder(self.embedder(x));

        # Set pooling operation for set
        #self.pooler = torch.max; #torch.mean;
        self.pooler = torch.sum; #torch.mean;

        # Initialize the parameters with xavier
        modules = ['embedder', 'postEmbedder', 'imgTransform', 'combine'];
        modules = [getattr(self, mod) for mod in modules if hasattr(self, mod)];
        initializeWeights(modules, 'xavier');

    # function to score the instance with set
    def scoreInstanceSet(self, instEmbed, setEmbed):
        # Expand instEmbed accordingly
        if instEmbed.dim() == 3:
            # Expand set and concat
            setEmbed = setEmbed.unsqueeze(1).expand(setEmbed.size(0), \
                                                    instEmbed.size(1), \
                                                    setEmbed.size(1))
            catEmbed = torch.cat((instEmbed, setEmbed), 2);
            return bottle(self.combine, catEmbed);
        else:
            catEmbed = torch.cat((instEmbed, setEmbed), 1);
            return self.combine(catEmbed);

    def trainStep(self, dataloader):
        # Grab a train batch
        batch = dataloader.getTrainBatch();

        # Extract set, positive and negative instance
        setEmbed = bottle(self.wordTransform, Variable(batch['set']));
        # if set is empty, reset to zero
        # TODO: this is bad! Breaks the history, works only for w2v models
        if self.setSize == 0: setEmbed.data.fill_(0.0);

        setEmbed = self.pooler(setEmbed, 1);
        if type(setEmbed).__name__ == 'tuple': setEmbed = setEmbed[0];
        setEmbed = setEmbed.squeeze();

        posEmbed = bottle(self.wordTransform, Variable(batch['pos']));
        negEmbed = bottle(self.wordTransform, Variable(batch['neg']));

        # If image exists
        if 'image' in batch:
            imgEmbed = self.imgTransform(Variable(batch['image']));
            setEmbed = torch.cat((setEmbed, imgEmbed), 1);

        # Interact positive and negative instances to get score
        posScore = self.scoreInstanceSet(posEmbed, setEmbed).view(-1);
        negScore = self.scoreInstanceSet(negEmbed, setEmbed).view(-1);

        # Get target labels (1's)
        labels = Variable(self.labels.expand_as(posScore.data));
        loss = self.criterion(posScore, negScore, labels);

        # Computes gradient for all the variables
        loss.backward();

        return loss.data[0];

    # Evaluate on a given set, also save top 10 words
    def evaluate(self, dataloader, dtype):
        # network in evaluation mode
        self.eval();
        gtRanks = [];
        numInst = dataloader.numInst[dtype];

        # save all scores and gtLabels
        scores = [];
        gtLabels = [];
        imageIds = [];

        # Get gt scores for all options
        for startId in progressbar(range(0, numInst, self.batchSize)):
            # Obtain test batch, argument set and GT members
            batch = dataloader.getTestBatch(startId, dtype);
            batchSize = batch['set'].size(0);

            # Extract set, positive
            setEmbed = bottle(self.wordTransform, Variable(batch['set']));
            # if set is empty, reset to zero
            if self.setSize == 0: setEmbed.data.fill_(0.0);

            setEmbed = self.pooler(setEmbed, 1);
            if type(setEmbed).__name__ == 'tuple': setEmbed = setEmbed[0];
            setEmbed = setEmbed.squeeze();

            # If image exists
            if 'image' in batch:
                imgEmbed = self.imgTransform(Variable(batch['image']));
                setEmbed = torch.cat((setEmbed, imgEmbed), 1);

            # current batch scores
            batchScores = torch.FloatTensor(batchSize, self.vocabSize);

            # Get the scores for all possible options
            for ii in range(0, self.vocabSize, self.batchSize):
                end = min(ii + self.batchSize, self.vocabSize);

                # Interact gt and set to get score
                argInds = torch.arange(ii, end).long().unsqueeze(0);
                if self.useGPU: argInds = argInds.cuda();
                argInds = argInds.repeat(batchSize, 1);
                argEmbed = bottle(self.wordTransform, Variable(argInds));
                argScore = self.scoreInstanceSet(argEmbed, setEmbed);
                # save scores for this batch
                batchScores[:, ii:end] = argScore.data.float();

            # Assign the set least possible score (-Inf) to set elements
            rangeInds = torch.arange(0, batchSize).long();
            for ii in range(self.evalSize):
                # satwik: edits for new pytorch
                scatInds = torch.stack((rangeInds, batch['set'][:, ii].cpu()), 1);
                batchScores.scatter_(1, scatInds, float('-inf'));

            # Convert to numpy array
            batchScores = batchScores.numpy();
            # rank data is ascending, need descending
            batchRanks = np.apply_along_axis(rankdata, 1, -1*batchScores);
            # save the batch scores
            scores.append(batchScores);

            # Assign the ranks
            gtLabels.extend(batch['pos']);
            if 'imageId' in batch: imageIds.extend(batch['imageId']);
            for ii in range(batchSize):
                gtRank = [batchRanks[ii, jj] for jj in batch['pos'][ii]];
                gtRanks.append(gtRank);

        # Compute rank statistics
        metrics = computeRankStats(np.concatenate(gtRanks));
        # network in training mode
        self.train();

        return metrics, np.concatenate(scores), {'gtLabels': gtLabels, \
                                                'imageId': imageIds};

    # Initialize word embeddings
    def initEmbeddings(self, (embeds, inds)):
        weight = self.embedder.weight.data;
        # Assign embeds
        for row in inds: weight[row] = torch.from_numpy(embeds[row]);
        # Copy back
        self.embedder.weight.data.copy_(weight);

    # Save model
    def save(self, path):
        content = {'combine': self.combine,\
                    'embedder': self.embedder, \
                    'postEmbedder': self.postEmbedder, \
                    'pooler': self.pooler, \
                    'params': self.params};
        if hasattr(self, 'imgTransform'):
            content['imgTransform'] = self.imgTransform;
        torch.save(content, path);

    # Load model
    def load(self, loadPath):
        content = torch.load(loadPath);

        flags = ['params', 'embedder', 'postEmbedder', 'combine', \
                'pooler', 'imgTransform'];

        for flag in flags:
            if flag in content: setattr(self, flag, content[flag]);
        print(self.postEmbedder)
        # initialize alias for wordTransform
        self.wordTransform = lambda x: self.postEmbedder(self.embedder(x));

