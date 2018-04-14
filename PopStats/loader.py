import pdb
import h5py
import numpy as np
from tqdm import tqdm, trange

class DataIterator(object):
    def __init__(self, fname, batch_size, shuffle=False):

        self.fname = fname
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load data from Matlab
        with h5py.File(fname) as f:
            #pdb.set_trace()
            self.L = np.asscalar(f['L'][()].astype('int32'))
            self.N = np.asscalar(f['N'][()].astype('int32'))
            self.t = np.squeeze(f['X_parameter'][()])
            self.X = f['X'][()]
            self.d = self.X.shape[0]
            self.X = np.reshape(self.X.transpose(), [-1,self.N,self.d]).astype('float32')
            self.y = np.squeeze(f['Y'][()]).astype('float32')
            #self.y = 3*(self.y + 74.74)
        
        assert len(self.y) >= self.batch_size, \
            'Batch size larger than number of training examples'
            
    def __len__(self):
        return len(self.y)//self.batch_size

    def get_iterator(self, loss=0.0):
        if self.shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(rng_state)
            np.random.shuffle(self.y)
            np.random.set_state(rng_state)
            np.random.shuffle(self.t)
        return tqdm(self.next_batch(),
                    desc='Train loss: {:.4f}'.format(loss),
                    total=len(self), mininterval=1.0, ncols=80)
                    
    def next_batch(self):
        start = 0
        end = self.batch_size
        while end <= self.L:
            yield self.X[start:end], self.y[start:end]
            start = end
            end += self.batch_size
