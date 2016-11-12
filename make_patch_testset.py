#!/usr/bin/env python
'''
 create a val/test set for a patch task
 the train examples are generated on the fly, but the val set should be fixed
 so we use this tool to generate it.
 uses the same hdf5 format also used by the training routine
 it can be generated with ./extract_crops_from_detections.py
'''

from __future__ import print_function
import sys, os, time
import os.path as path
from argparse import ArgumentParser
from progressbar import ProgressBar
from datetime import datetime
import concurrent.futures
import h5py

import ipdb

import numpy as np

from matplotlib import pyplot as plt

from pUtils import git, dump_args, recreate_dir

from tasks import TaskPatchesCached, TaskPairwisePatches

def main(args):
    print('read mean&std file...')
    h = h5py.File(args.mean_file)
    patch_mean = h['mean'][:].astype(np.float32) # this is in [0,1] format
    patch_std  = h['std'][:].astype(np.float32)

    if args.no_std_scaling:
        print('will only shift not standard scale the examples')
        patch_std = None

    if args.pairwise:
        task = TaskPairwisePatches(filename_crops=args.input,
                                   patch_size=32,
                                   layout=np.array([3, 6]),
                                   jitter=4,
                                   drop_colorchannel=False,
                                   patch_mean=patch_mean,
                                   patch_std=patch_std)
    else:
        if args.patch_task == 'pt18':
            pt_layout = np.array([3, 6])
        else:
            pt_layout = np.array([2, 4])
        task = TaskPatchesCached(filename_crops=args.input,
                                 patch_size=32,
                                 layout=pt_layout,
                                 jitter=4,
                                 drop_colorchannel=False,
                                 patch_mean=patch_mean,
                                 patch_std=patch_std)

    crops = np.zeros((task.patches_per_call * args.n_examples, 3, 32, 32), dtype=np.float32)
    labels = np.zeros(args.n_examples)
    with ProgressBar(max_value=args.n_examples) as progress:
        for idx, r in enumerate([task(x % task.N_crops) for x in range(args.n_examples)]):
            x, y = r
            if isinstance(x, tuple):
                if len(x) == 2:
                    crops[2*idx, :] = x[0]
                    crops[2*idx+1, :] = x[1]
                else:
                    raise ValueError('we currently can only handle pairwise patch tasks...')
            else:
                crops[idx, :] = x
            labels[idx] = y
            progress.update(idx)

    h = h5py.File(args.output, 'w')
    h.create_dataset('patches', data=crops)
    h.create_dataset('labels', data=labels)
    h.close()
    print('max label: ', np.max(labels))
    print('min label: ', np.min(labels))
    print('patch shape: ', crops.shape)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y_%m_%d__%H:%M:%S')
    pid = os.getpid()

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--patch_size', '-p', type=int, default=32)
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--patch_task', default='pt18', choices=('pt18', 'pt8'))

    parser.add_argument('--no-std-scaling', action='store_true',
                        help='only apply shift to training examples, but not scaling by standard deviation')

    parser.add_argument('--n_examples', '-n', type=int, default=5000,
                        help='# of (patch, label) pairs to extract')
    parser.add_argument('input', help='hdf5 file with detection crops')
    parser.add_argument('mean_file', help='hdf5 file with mean and std-deviation of training patches')
    parser.add_argument('output', help='hdf5 of preprocessed val/test data')

    args = parser.parse_args()
    main(args)
