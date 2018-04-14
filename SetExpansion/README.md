## Requirements
* Python 2
* PyTorch
* tqdm
* h5py
* NumPy
* SciPy
* gensim
* html

## Data

For the text set expansion, we are including the data in the `data` folder. It contains the 50 top words from each topic coming from a LDA trained on the English Wikipedia. We provide splits of TRAIN, VALID, TEST for case 1k, 3k, and 5k topics.

## Experiments

To train the models from scratch, you may simply run:

    ./train_text.sh <GPU ID>

There are following parameters which can be set in `train_text.sh`:

```bash
# The margin used in Margin Loss to train
MARGIN=0.1

# Type of model
# - w2v_sum corresponds to DeepSets
MODEL='w2v_sum'

# Size of embedding
embedSize=50

# Eval size: During inference the size of sets to use
evalSize=5

# Learning rate of optimizer
learningRate=0.0001

# Number of epochs
numEpochs=10000
```

For more options and their details, please visit `options.py`.

### Pretrained models

We also provide pretrained models for each of the three cases in `models`. To replicate results using the pretrained model, you may simply run:

    ./eval_trained_model.sh <GPU ID>
  
