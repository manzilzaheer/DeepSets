# Script to load data and generate batches
# author: satwik kottur
import json
import numpy as np
import pdb
import h5py
import random
import torch

class AbstractLoader:
    # Get train batch
    def getTrainBatch(self): raise NotImplementedError();

    # Get test batch
    def getTestBatch(self, startId, dtype): raise NotImplementedError();

    # Get a single instance
    #@abstractmethod
    #def getIndexInstance(self, index, dtype):
    #    return self.data[dtype][index];
