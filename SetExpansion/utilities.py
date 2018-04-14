# Contains various utilities
from gensim.models import word2vec
import torch
import numpy as np
import pdb
import math
import sys
sys.path.append('scripts/');
from html import HTML
import gensim

# Handles 2D and 3D, reduces one dimension
def bottle(module, inputData):
    if inputData.dim() == 2:
        output = module(inputData.view(-1));
    elif inputData.dim() == 3:
        output = module(inputData.view(-1, inputData.size(2)));

    # Get it back to original shape
    return output.view(inputData.size(0), inputData.size(1), -1);

# Computing rank statistics
def computeRankStats(ranks, verbose=True):
    # Handle both numpy and torch
    if type(ranks).__module__ == np.__name__: ranks = torch.FloatTensor(ranks);

    # Check if any of the ranks are zeros
    assert ranks.eq(0.0).sum() == 0, 'Few of the ranks are zeros!';

    ranks = ranks.view(-1);
    # metrics:
    # recall@10,100,1000; mean, median ranks, mean reciprocal rank
    stats = {};
    stats['mean'] = ranks.mean();
    stats['median'] = ranks.median();
    stats['mrr']  = ranks.reciprocal().mean();
    stats['r10'] = 100 * torch.sum(ranks.le(10))/float(ranks.size(0));
    stats['r100'] = 100 * torch.sum(ranks.le(100))/float(ranks.size(0));
    stats['r1000'] = 100 * torch.sum(ranks.le(1000))/float(ranks.size(0));

    # pretty print
    if verbose:
        order = ['r10', 'r100', 'r1000', 'median', 'mean', 'mrr'];
        print('\n')
        for key in order:
            try: print('\t%s: %f' % (key, stats[key]))
            except: pdb.set_trace();

    return stats;

# Pre loading w2v embeddings
def composeEmbeddings(dl):
    # Embed all the words using word2vec
    model = gensim.models.KeyedVectors.load_word2vec_format(dl.embedPath, binary=True);
    #model = word2vec.Word2Vec.load_word2vec_format(dl.embedPath, binary=True);

    # For each set, get NN
    featSize = model['king'].shape[0];
    assert dl.embedSize == featSize, 'Inconsistent embed sizes!';

    # Copy the embeds
    embeds = np.zeros((dl.vocabSize, dl.embedSize));
    inds = [];
    for word, ind in dl.word2ind.iteritems():
        # note which words has word2vec embeddings
        if word in model:
            embeds[ind] = model[word];
            inds.append(ind);

    return embeds, inds;

# Compute scores
def rankBatchVersusGT(argScore, gtScore):
    # Expand the argument score, according to gt score
    argScore = argScore.transpose(1, 2).expand(argScore.size(0),\
                                                gtScore.size(1),\
                                                argScore.size(1));
    batchRanks = argScore.data.gt(gtScore.expand_as(argScore).data);
    ranks = batchRanks.sum(2).squeeze().float();
    return ranks;

# Initializing weights
def initializeWeights(moduleList, itype):
    assert itype=='xavier', 'Only Xavier initialization supported';

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, '_modules') and len(module._modules) > 0:
            # Iterate again
            initializeWeights(module, itype);
        else:
            # Initialize weights
            name = type(module).__name__;
            # If linear or embedding
            if name == 'Embedding' or name == 'Linear':
                fanIn = module.weight.data.size(0);
                fanOut = module.weight.data.size(1);

                factor = math.sqrt(2.0/(fanIn + fanOut));
                weight = torch.randn(fanIn, fanOut) * factor;
                module.weight.data.copy_(weight);

            # Check for bias and reset
            if hasattr(module, 'bias'): module.bias.data.fill_(0.0);

# Save top words
def saveTopWords(result, dataloader, dtype, topN = 10):
    # local aliases
    _, scores, gtRanks = result;

    # Create a page with 3 columns
    page = HTML(3);
    page.setTitle(['Set', 'Ground Truth', 'Top Words']);

    numInst = dataloader.numInst[dtype];
    for ii in xrange(numInst):
        rowContent = []; # row

        data = dataloader.getIndexInstance(ii, dtype);
        # set
        setData = data[:dataloader.evalSize];
        setWords = [dataloader.ind2word[str(setData[jj])] \
                        for jj in xrange(setData.size(0))];
        rowContent.append('\n'.join(setWords));

        # gt words, scores, ranks
        gtData = data[dataloader.evalSize:];
        gtWords = [dataloader.ind2word[str(gtData[jj])] \
                        for jj in xrange(gtData.size(0))];
        gtInfo = ['%s \t(%f)\t[%d]' \
                    % (gtWords[jj], scores[ii, gtData[jj]], gtRanks[ii, jj])\
                    for jj in xrange(gtData.size(0))];
        rowContent.append('\n'.join(gtInfo));

        # topN words, scores, ranks
        argScores = scores[ii].numpy();
        topData = argScores.argsort()[-topN:][::-1];
        topWords = [dataloader.ind2word[str(topData[jj])] \
                        for jj in xrange(topData.shape[0])];
        topInfo = ['%s \t(%f)\t[%d]' \
                    % (topWords[jj], scores[ii, topData[jj]], jj)\
                    for jj in xrange(topData.shape[0])];
        rowContent.append('\n'.join(topInfo));

        page.addRow(rowContent);

    # render page and save
    page.savePage(dataloader.resultPath);

# Visualizing the batch
# given a batch, collect image ids, words and display
def visualizeBatch(dataloader):
    # local alias
    dl = dataloader;

    # get batch
    batch = dl.getTrainBatch();
    # create a html page
    page = HTML(4);
    imgPath = 'train2014/COCO_train2014_%012d.jpg';

    # Get the unique image locations
    imgSum = batch['image'].sum(1).numpy();
    curSum = imgSum[0];
    count = 0;
    for ii in xrange(imgSum.shape[0]):
        if curSum != imgSum[ii]: count += 1;
        curSum = imgSum[ii];

        # New row
        # add image, set, pos, neg examples
        row = [page.linkImage(imgPath % batch['imageId'][count])];
        setWords = [dl.ind2word[jj] for jj in list(batch['set'][ii])];
        row.append(', '.join(setWords));

        row.append(dl.ind2word[batch['pos'][ii, 0]]);
        row.append(dl.ind2word[batch['neg'][ii, 0]]);

        # add the row
        page.addRow(row);

    # render page
    page.savePage('visualize_batch.html');

# evaluation for tagging - per tag evaluation
# adapted: fast tag code
def evaluateTagging(scores, gtLabels, topAnswers=5, verbose=True):
    #print('Number of tags: %d' % np.sum([len(ii) for ii in gtLabels]))
    numInst = scores.shape[0];
    vocabSize = scores.shape[1];

    # get the topAnswers number of predictions for each image
    predTags = np.argsort(scores)[:, -topAnswers:];
    predMat = np.zeros((numInst, vocabSize));
    gtMat = np.zeros((numInst, vocabSize));
    for ii in range(numInst):
        # gt matrix
        for word in gtLabels[ii]:
            gtMat[ii, word] = 1;

        # prediction matrix
        for jj in xrange(topAnswers):
            predMat[ii, predTags[ii, jj]] = 1;

    # precision
    eps = 2.2204e-16;
    gtPredProduct = np.sum(np.multiply(gtMat, predMat), 0);
    precisionDen = np.maximum(np.sum(predMat, 0), eps);
    precision = np.divide(gtPredProduct, precisionDen);
    meanPrecision = np.mean(precision);

    # recall
    recallDen = np.maximum(np.sum(gtMat, 0),  eps);
    recall = np.sum(np.multiply(gtMat, predMat), 0);
    recall = np.divide(gtPredProduct, recallDen);
    meanRecall = np.mean(recall);

    # f1 score
    #f1score =  2 * np.multiply(precision, recall);
    #f1scoreDen = np.max(precision + recall, 1e-8);
    #f1score = np.divide(f1score, f1scoreDen);
    if meanPrecision == 0 or meanRecall == 0: f1score = 0.0;
    else: f1score = 2*meanPrecision*meanRecall / (meanPrecision+meanRecall);

    # N+
    recallPositive = np.sum(gtPredProduct > 0);

    # pretty print output
    if verbose:
        print('\n\tPrecision: %f' % meanPrecision)
        print('\tRecall: %f' % meanRecall)
        print('\tF1 Score: %f' % f1score)
        print('\tN+: %f\n' % recallPositive)

    metrics = {};
    metrics['precision'] = meanPrecision;
    metrics['recall'] = meanRecall;
    metrics['f1score'] = f1score;
    metrics['positiveRecall'] = recallPositive;

    return metrics;

# Save the predicted tags along with gt
def savePredictedTags(scores, groundTruth, dataloader, topN=20):
    # local aliases
    gtLabels, imgIds = groundTruth['gtLabels'], groundTruth['imageId'];

    # Create a page with 3 columns
    page = HTML(3);
    page.setTitle(['Image', 'Ground Truth', 'Predicted Tags']);
    imgPath = 'val2014/COCO_val2014_%012d.jpg';
    numImgs = 100;

    for ii in xrange(numImgs):
        rowContent = [page.linkImage(imgPath % imgIds[ii])];
        #data = dataloader.getIndexInstance(ii, 'test');
        # set
        #setData = data[:dataloader.evalSize];
        #setWords = [dataloader.ind2word[str(setData[jj])] \
        #                for jj in xrange(setData.size(0))];
        #rowContent.append('\n'.join(setWords));

        # gt words, scores, ranks
        #gtData = data[dataloader.evalSize:];
        gtWords = [dataloader.ind2word[jj] for jj in gtLabels[ii]];
        rowContent.append('\n'.join(gtWords));

        # Get the predicted tags
        imgScore = scores[ii, :];
        predTags = imgScore.argsort()[-topN:][::-1];
        tags = [dataloader.ind2word[jj] for jj in predTags];
        rowContent.append('\n'.join(tags));

        page.addRow(rowContent);

    # render page and save
    page.savePage('img_tags_espgame.html');
