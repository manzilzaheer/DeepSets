## Requirements

* Python 2
* MATLAB
* PyTorch
* h5py
* numpy
* tqdm

## Generating the data

In order to generate the data required for these experiments, run the commands below in this directory. 

```
cd generator
./generate.sh
```

It should take a while for the data to be generated for all experiments.

## Running the experiments

There are 4 tasks as in the paper and for each task, 11 different-sized training sets. You may run an experiment using a particular training set by running the command below:

```
python2 trainer.py <task_id> <exp_id>
```

where `<task_id>` is the ID of the task you would like to run, ranging from 1 to 4. You may find more information on task descriptions in the first few lines of the files `generator/generate_task<task_id>_dataset.m`.

`<exp_id>` corresponds to the training set size, ranging from 7 to 17, in particular, there are `2**<exp_id>` samples in each dataset.

**Note:** There will be **_no_** output on stdout. All the logs and model outputs will be saved in to `generator/data/task<task_id>/exp<exp_id>`.
