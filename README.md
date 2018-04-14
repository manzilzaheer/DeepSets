# DeepSets

Code and models for the paper [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets)

We study the problem of designing models for machine learning tasks defined on sets. In contrast to the traditional approach of operating on fixed dimensional vectors, we consider objective functions defined on sets and are invariant to permutations. Such problems are widespread, ranging from the estimation of population statistics, to anomaly detection in piezometer data of embankment dams, to cosmology. Our main theorem characterizes the permutation invariant objective functions and provides a family of functions to which any permutation invariant objective function must belong. This family of functions has a special structure which enables us to design a deep network architecture that can operate on sets and which can be deployed on a variety of scenarios including both unsupervised and supervised learning tasks. We demonstrate the applicability of our method on population statistic estimation, point cloud classification, set expansion, and outlier detection.

## Code Structure

Here we provide code to replicate some of the experiments reported in the paper. Please look at `README` for individual experiments.

```
/
├── DigitSum: Sum of Set of Digits (Text/Image)
│   
├── PointClouds: Classification of Point Cloud Data
│   
├── PopStats: Estimation population statistics from iid sample set of data
│   
├── SetExpansion: For expanding a given set from a large pool of candidates
```

## Citation
If you use this code, please cite our paper
```
@incollection{deepsets,
title = {Deep Sets},
author = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Ruslan R and Smola, Alexander J},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {3391--3401},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/6931-deep-sets.pdf}
}
```
