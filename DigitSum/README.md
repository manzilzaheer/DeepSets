## Acquiring the data

The text portion of this experiments needs no data.

The image portion of this experiment uses InfiMNIST digits (http://leon.bottou.org/projects/infimnist). A processed version of MNIST is used for these experiments. One can use the provided `generate.sh` to get binary files. Then one can utilize the excellent project from Congrui Yi https://github.com/CY-dev/infimnist-parser to load the binaries into python as uint8 numpy array. Split the digits into 8 partitions of roughly equal sizes. For each partition, save in a binary file N as int32 followed by D as int32 followed by N\*D pixel values as int32 as well. Here N denotes the size of the partition and D denotes the image size which should be 28x28=784 for MNIST. Name these binary files `mnist8m_0_features.bin`, `mnist8m_1_features.bin`, ..., `mnist8m_7_features.bin`. Save the labels similarly, i.e. in a binary file N as int32 followed by D as int32 followed by N labels as int32 as well. Here N denotes the size of the partition and D should be 1. them as (N, 784) dimensional. Name these binary files `mnist8m_0_labels.bin`, `mnist8m_1_labels.bin`, ..., `mnist8m_7_labels.bin`. The ipython notebook assumes availability of these files. Please contact us if you are having difficulty in generating the datasets.

## Running the experiments

You may follow along the two notebooks for experiments. In order to replicate the results, you may need to perform a bit of parameter tuning.
