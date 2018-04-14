# Contains model descriptions
# author: satwik kottur
import torch
import torch.nn as nn
import pdb
from miscModules import *

def selectModel(params):
    embedder, postEmbedder, imgTransform, combine = None, None, None, None;
    embedder = nn.Embedding(params['vocabSize'], params['embedSize']);
    postEmbedder = Identity();
    ###########################################################################
    if params['modelName'] == 'regular':
        postEmbedder = nn.Sequential(
                    nn.Dropout(params['dropout']),
                    nn.ReLU(),
                    nn.Linear(params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    );
        combine = nn.Sequential(
                    nn.Linear(2*params['hiddenSize'], params['hiddenSize']),
                    nn.Dropout(params['dropout']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], 1),
                    nn.Tanh(),
                    );

    ###########################################################################
    elif params['modelName'] == 'w2v_only':
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.Dropout(params['dropout']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], 1),
                    nn.Tanh(),
                    );

    ###########################################################################
    elif params['modelName'] == 'w2v_freeze':
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    nn.Sigmoid(),
                    );
    ###########################################################################
    elif params['modelName'] == 'w2v_freeze_tanh':
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.Tanh(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.Tanh(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    nn.Sigmoid(),
                    );
    ###########################################################################
    elif params['modelName'] == 'w2v_freeze_sum':
        combine = nn.Sequential(
                    SplitSum(params['embedSize']),
                    nn.Linear(params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    nn.Sigmoid(),
                    );
    ###########################################################################
    elif params['modelName'] == 'w2v_max':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    SplitMax(params['hiddenSize']),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
    ###########################################################################
    elif params['modelName'] == 'w2v_sum':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.Tanh(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                        nn.Tanh(),
                    );
        combine = nn.Sequential(
                    SplitSum(params['hiddenSize']),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    nn.Tanh(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.Tanh(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
    ###########################################################################
    elif params['modelName'] == 'w2v_concat':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    nn.Linear(2*params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
    ###########################################################################
    elif params['modelName'] == 'coco_w2v_simple':
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], 1),
                    #nn.ReLU(),
                    nn.Sigmoid(),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            nn.Dropout(params['dropout']),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            );
    ###########################################################################
    elif params['modelName'] == 'coco_w2v':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    nn.Linear(2*params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['hiddenSize']),
                            nn.Dropout(params['dropout']),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['hiddenSize']),
                            );
    ###########################################################################
    elif params['modelName'] == 'tagger_sum':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    Multimodal([params['hiddenSize'], 2 * params['hiddenSize']]),
                    nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.Dropout(params['dropout']),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, params['hiddenSize']),
                            nn.ReLU(),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, params['hiddenSize']),
                            nn.ReLU(),
                            );
    ###########################################################################
    elif params['modelName'] == 'tagger_sum_large':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    Multimodal([params['hiddenSize'], 2 * params['hiddenSize']]),
                    nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.Dropout(params['dropout']),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, 2 * params['hiddenSize']),
                            nn.ReLU(),
                            nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                            nn.ReLU(),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, 2 * params['hiddenSize']),
                            nn.ReLU(),
                            nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                            nn.ReLU(),
                            );
    ###########################################################################
    elif params['modelName'] == 'tagger_sum_sum_large':
        postEmbedder = nn.Sequential(
                        nn.Linear(params['embedSize'], params['hiddenSize']),
                        nn.ReLU(),
                        nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    );
        combine = nn.Sequential(
                    MultimodalSum([params['hiddenSize'], 2 * params['hiddenSize']]),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.Dropout(params['dropout']),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, 2 * params['hiddenSize']),
                            nn.ReLU(),
                            nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                            nn.ReLU(),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['featSize']/2),
                            nn.ReLU(),
                            nn.Linear(params['featSize']/2, 2 * params['hiddenSize']),
                            nn.ReLU(),
                            nn.Linear(2 * params['hiddenSize'], params['hiddenSize']),
                            nn.ReLU(),
                            );
    ###########################################################################
    elif params['modelName'] == 'coco_w2v_sum':
        combine = nn.Sequential(
                    SplitSum(params['embedSize']),
                    nn.Linear(params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    #nn.ReLU(),
                    nn.Sigmoid(),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            nn.Dropout(params['dropout']),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            );
    ###########################################################################
    elif params['modelName'] == 'coco_w2v_linear':
        postEmbedder = nn.Linear(params['embedSize'], params['embedSize']);
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], params['hiddenSize']/2),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize']/2, 1),
                    #nn.ReLU(),
                    nn.Sigmoid(),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            nn.Dropout(params['dropout']),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            );
    ###########################################################################
    elif params['modelName'] == 'espgame_w2v_linear':
        postEmbedder = nn.Linear(params['embedSize'], params['embedSize']);
        combine = nn.Sequential(
                    nn.Linear(2*params['embedSize'], params['hiddenSize']),
                    nn.ReLU(),
                    nn.Linear(params['hiddenSize'], 1),
                    nn.Sigmoid(),
                    );
        if params['dropout'] > 0:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            nn.Dropout(params['dropout']),
                            );
        else:
            imgTransform = nn.Sequential(
                            nn.Linear(params['featSize'], params['embedSize']),
                            );
    ###########################################################################
    content = {'postEmbedder': postEmbedder, \
                'combine': combine, \
                'embedder': embedder};
    if imgTransform is not None: content['imgTransform'] = imgTransform;
    return content;
