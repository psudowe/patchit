# PatchIt Scripts

This is a subset of the scripts used in the experiments for [our BMVC'16 paper](http://www.vision.rwth-aachen.de/publication/00140/).
The purpose of this code is mainly to illustrate the `PatchTask` procedure and get others started with it.


The experiments in [the paper](http://www.vision.rwth-aachen.de/publication/00140/) are based on the [Parse27k dataset](http://www.vision.rwth-aachen.de/parse27k).
The `PatchTask` method allows researchers and developers to pretrain a network in a semi-supervised way.
We also provide the unlabeled person detections from the full training video sequences on the [Parse27k dataset website](http://www.vision.rwth-aachen.de/parse27k).



If you use this code, please cite our paper:
```
@InProceedings{Sudowe16BMVC,
    author = {Patrick Sudowe and Bastian Leibe},
	title  = {{PatchIt: Self-Supervised Network Weight Initialization for Fine-grained Recognition}},
	booktitle = BMVC,
	year = {2016}
}
```


##  Dependencies

We use Python 2.7 on a 64-bit Linux -- but the tools should run on other platforms as well.

The scripts have some dependencies. If you are using Python regularly for scientific computing, 
all of this is likely installed already. Otherwise they can all be easily obtained through `pip`.
These tools depend (directly or indirectly) on:

```
futures
progressbar2
h5py
Pillow
scipy
scikit-image
matplotlib
theano
lasagne
```

##  Usage

The main purpose of these scripts is to further illustrate the description of the `PatchTask` method.
The code provided here does not fully reproduce all experiments.

* `patch_training.py`: the script to run a `PatchTask` training
* `nets.py`: specification of the network architectures to be used (add your own net here)
* `tasks.py`: read this for the details of the `PatchTask`


## Bugs & Problems & Updates
The tools here were taken from a larger set of tools as we prepared the dataset for publication.
In case we notice problems or bugs introduced in this process, we will fix these here.

If you are working with the *Parse-27k* dataset, we encourage you to *follow this repository*.
In case there are any bug-fixes or changes to the evaluation pipeline, GitHub will let you know.

