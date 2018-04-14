# contains misc modules use to construct models
# author: satwik kottur

import torch
import torch.nn as nn
import pdb

# identity module
class Identity(nn.Container):
    def __init__(self):
        super(Identity, self).__init__();

    def forward(self, inTensor):
        return inTensor;

# module to split at a given point and sum
class SplitSum(nn.Container):
    def __init__(self, splitSize):
        super(SplitSum, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize];
            secondHalf = inTensor[:, self.splitSize:];
        else:
            firstHalf = inTensor[:, :, :self.splitSize];
            secondHalf = inTensor[:, :, self.splitSize:];
        return firstHalf + secondHalf;

# module to split at a given point and sum
class Multimodal(nn.Container):
    def __init__(self, splitSize):
        super(Multimodal, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            instEmbed = inTensor[:, :self.splitSize[0]];
            setEmbed = inTensor[:, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, self.splitSize[1]:];
            concatDim = 1;
        else:
            instEmbed = inTensor[:, :, :self.splitSize[0]];
            setEmbed = inTensor[:, :, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, :, self.splitSize[1]:];
            concatDim = 2;

        return torch.cat((setEmbed + instEmbed, imgEmbed), concatDim);

# module to split at a given point and sum
class MultimodalSum(nn.Container):
    def __init__(self, splitSize):
        super(MultimodalSum, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            instEmbed = inTensor[:, :self.splitSize[0]];
            setEmbed = inTensor[:, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, self.splitSize[1]:];
        else:
            instEmbed = inTensor[:, :, :self.splitSize[0]];
            setEmbed = inTensor[:, :, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, :, self.splitSize[1]:];

        return setEmbed + instEmbed + imgEmbed;

# module to split at a given point and max
class SplitMax(nn.Container):
    def __init__(self, splitSize):
        super(SplitMax, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize];
            secondHalf = inTensor[:, self.splitSize:];
        else:
            firstHalf = inTensor[:, :, :self.splitSize];
            secondHalf = inTensor[:, :, self.splitSize:];
        numDims = firstHalf.dim();
        concat = torch.cat((firstHalf.unsqueeze(numDims), \
                            secondHalf.unsqueeze(numDims)), numDims);
        maxPool = torch.max(concat, numDims)[0];
        # satwik: edits for older pytorch version
        #maxPool = torch.max(concat, numDims)[0].squeeze(numDims);
        return maxPool;

# Module to nullify the inputs, ie fill them with zeros
class Nullifier(nn.Container):
    def __init__(self):
        super(Nullifier, self).__init__();

    def forward(self, inTensor):
        outTensor = inTensor.clone();
        outTensor.fill_(0.0);
        return outTensor;
