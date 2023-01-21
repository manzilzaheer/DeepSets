# Script to read options
# author: satwik kottur
import argparse
import pdb
from time import gmtime, strftime
import os

def readOptions():
    parser = argparse.ArgumentParser(description='Training deepsets');

    parser.add_argument('-inputData', required=True, \
                        help='Input data path');
    parser.add_argument('-savePath', default='models/', \
                        help='Model save path');
    parser.add_argument('-modelName', default='basic', \
                        help='label for model');
    parser.add_argument('-embedPath', default='', \
                        help='preloading word embeddings');
    parser.add_argument('-resultPath', default='', \
                        help='storing results during evaluation');
    parser.add_argument('-loadPath', default='', \
                        help='load a saved model for evaluation');
    parser.add_argument('-dataset', required=True, \
                        help='kind of dataset to use');

    # Model parameters
    parser.add_argument('-embedSize', default=300, type=int,\
                        help='Embed size for words');
    parser.add_argument('-hiddenSize', default=150, type=int,\
                        help='Embed size for words');
    parser.add_argument('-evalSize', default=4, type=int,\
                        help='Evaluation set size');
    parser.add_argument('-setSize', default=-1, type=int,\
                        help='Word set size (-1 picks at random)');
    parser.add_argument('-margin', default=0.3, type=float,\
                        help='Margin for the loss function');
    parser.add_argument('-dropout', default=0.0, type=float,\
                        help='Dropout between layers');

    # Optimization options
    parser.add_argument('-batchSize', default=100, type=int,\
                        help='Batch size (adjust on GRAM)')
    parser.add_argument('-numEpochs', default=400, type=int,\
                        help='Maximum number of epochs to run')
    parser.add_argument('-evalPerEpoch', default=5, type=int,\
                        help='Number of epochs after which evaluate')
    parser.add_argument('-learningRate', default=1e-3, type=float,\
                        help='Initial learning rate')
    parser.add_argument('-minLR', default=1e-5, type=float,\
                        help='Minimum learning rate')
    parser.add_argument('-useGPU', default=1, type=int,\
                        help='1 for GPU and 0 for CPU')
    parser.add_argument('-backend', default=True, type=bool,\
                        help='true for GPU and false for CPU')

    try: parsed = vars(parser.parse_args());
    except IOError: msg: parser.error(str(msg));

    # Adjust model name to append current time
    time = strftime("-%d-%b-%Y-%X/", gmtime());
    parsed['savePath'] += parsed['savePath'][:-1] + time;
    # Create the folder
    os.makedirs(parsed['savePath'])

    if parsed['useGPU'] == 0: parsed['useGPU'] = False;
    else: parsed['useGPU'] = True;
    if parsed['embedPath'] == '': parsed['embedPath'] = None;

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in parsed.iteritems(): print(fmtString % keyPair)
    return parsed;
