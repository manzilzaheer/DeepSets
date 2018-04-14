import os
import pdb
import sys
import h5py
import numpy as np
from tqdm import tqdm, trange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import tensorflow as tf

from loader import DataIterator
from model import DeepSet

def log_scalar(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)

    
class Trainer(object):

    def __init__(self, in_dims, truth, num_epochs=1000, out_dir='out/', log_dir='logs/'):
        self.truth = truth        
        
        self.model = DeepSet(in_dims).cuda()
        
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        
        self.optim = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=50, verbose=True)
        
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.out_dir = out_dir
        
    def fit(self, train, valid=None):
        train_loss = 0.0
        best_mae = 1.0e3
        best_mse = 1.0e6
        loss_val = 0.0

        train_writer = tf.summary.FileWriter(self.log_dir)

        for j in trange(self.num_epochs, desc="Epochs: ", ncols=80):
            train_iterator = train.get_iterator(train_loss)
            for X, y in train_iterator:
                #pdb.set_trace()
                self.optim.zero_grad()
                y_pred = self.model(Variable(torch.from_numpy(X)).cuda())
                loss = self.l2(y_pred, Variable(torch.from_numpy(y).cuda()))
                loss_val = np.asscalar(loss.data.cpu().numpy())
                train_loss = 0.9*train_loss + 0.1*loss_val
                loss.backward()
                self.optim.step()
                train_iterator.set_description(
                    'Train loss: {0:.4f}'.format(train_loss))

            test_mae, test_mse = self.evaluate(valid)
            if test_mae < best_mae:
                best_mae = test_mae
                torch.save(self.model, self.out_dir + 'best_mae_model.pth')
            if test_mse < best_mse:
                best_mse = test_mse
                torch.save(self.model, self.out_dir + 'best_mse_model.pth')
            self.scheduler.step(best_mse)

            tqdm.write(
                'After epoch {0} Test MAE: {1:0.6g} Best MAE: {2:0.6g} Test MSE: {3:0.6g} Best MSE: {4:0.6g} '
                'Train Loss: {5:0.6f}'.format(
                    j+1, test_mae, best_mae, test_mse, best_mse, train_loss))

            self.visualize(valid)

            log_scalar(train_writer, 'train_loss', train_loss, j+1)
            log_scalar(train_writer, 'test_mae', test_mae, j+1)
            log_scalar(train_writer, 'test_mse', test_mse, j+1)
            

        for j in trange(self.num_epochs, desc="Epochs: "):
            train_iterator = train.get_iterator(train_loss)
            for X, y in train_iterator:
                #pdb.set_trace()
                self.optim.zero_grad()
                y_pred = self.model(Variable(torch.from_numpy(X)).cuda())
                loss = self.l1(y_pred, Variable(torch.from_numpy(y).cuda()))
                loss_val = np.asscalar(loss.data.cpu().numpy())
                train_loss = 0.9*train_loss + 0.1*loss_val
                loss.backward()
                self.optim.step()
                train_iterator.set_description(
                    'Train loss: {0:.4f}'.format(train_loss))

            test_mae, test_mse = self.evaluate(valid)
            if test_mae < best_mae:
                best_mae = test_mae
                torch.save(self.model, self.out_dir + 'best_mae_model.pth')
            if test_mse < best_mse:
                best_mse = test_mse
                torch.save(self.model, self.out_dir + 'best_mse_model.pth')

            tqdm.write(
                'After epoch {0} Test MAE: {1:0.6g} Best MAE: {2:0.6g} Test MSE: {3:0.6g} Best MSE: {4:0.6g} '
                'Train Loss: {5:0.6f}'.format(
                    j+1, test_mae, best_mae, test_mse, best_mse, train_loss))

            self.visualize(valid)

            log_scalar(train_writer, 'train_loss', train_loss, self.num_epochs+j+1)
            log_scalar(train_writer, 'test_mae', test_mae, self.num_epochs+j+1)
            log_scalar(train_writer, 'test_mse', test_mse, self.num_epochs+j+1)
            
        return best_mae, best_mse        
    
    def evaluate(self, test):
        counts = 0
        sum_mae = 0.0
        sum_mse = 0.0
        test_iterator = test.get_iterator()
        for X, y in test_iterator:
            counts += 1
            y_pred = self.model(Variable(torch.from_numpy(X)).cuda())
            sum_mae += self.l1(y_pred, Variable(torch.from_numpy(y)).cuda()).data.cpu().numpy()
            sum_mse += self.l2(y_pred, Variable(torch.from_numpy(y)).cuda()).data.cpu().numpy()
        return np.asscalar(sum_mae/counts), np.asscalar(sum_mse/counts)
        
    def predict(self, test):
        y_preds = []
        for X, y in test.next_batch():
            y_pred = self.model(Variable(torch.from_numpy(X)).cuda())
            y_preds.append(y_pred.data.cpu().numpy())
        return np.concatenate(y_preds)
        
    def visualize(self, iterator):
        y_pred = self.predict(iterator)
        
        font = {'size': 14}
        matplotlib.rc('font', **font)
        scale = 0.5
        plt.figure(figsize=(10*scale, 7.5*scale))

        plt.plot(self.truth[0], self.truth[1])
        plt.plot(iterator.t, y_pred, 'x')
        plt.xlabel('Index')
        plt.ylabel('Statistics')
        plt.legend(['Truth', 'DeepSet'], loc=3, fontsize=12)
        plt.tight_layout()
        plt.savefig(self.out_dir + "current_status.png")
        plt.close()
                
        
if __name__ == '__main__':
    
    # Task id
    task_id = sys.argv[1]
    exp_id = sys.argv[2]

    ddir = 'generator/data/task{0}/'.format(task_id)
    odir = ddir+'exp{0}/'.format(exp_id)
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    f1 = open(odir + 'stdout.txt', 'w')
    f2 = open(odir + 'stderr.txt', 'w')
    sys.stdout = f1
    sys.stderr = f2
    print('Running Task {0} with L = 2^{1}'.format(task_id, exp_id))
    
    # Load dataset
    train = DataIterator(ddir+'data_{0}.mat'.format(exp_id), 128, shuffle=True)
    valid = DataIterator(ddir+'val.mat', 128)
    test  = DataIterator(ddir+'test.mat', 128)
    assert train.d == valid.d and train.d == test.d, \
            'Dimensions of train, valid, and test do not match!'

    # Load truth from Matlab
    with h5py.File(ddir+'truth.mat') as f:
        t = np.squeeze(f['X_parameter'][()])
        y = np.squeeze(f['Y'][()])
    print 'Loaded dataset'
    
    nb_epoch = 500 # np.max([1024*1024/train.L,100])
    print train.d
    t = Trainer(train.d, (t,y), nb_epoch, odir, odir+'logs/')
    a, b = t.fit(train, valid)
    t.model = torch.load(odir + 'best_mse_model.pth')
    a, b = t.evaluate(test)
    print('Test set evaluation: ')
    print('\t MAE: {0:0.6g} \t MSE: {1:0.6g}'.format(a, b))
    
    f1.close()
    f2.close()
        
