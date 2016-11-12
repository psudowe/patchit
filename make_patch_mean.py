#!/usr/bin/env python 

from __future__ import print_function
from argparse import ArgumentParser
from progressbar import ProgressBar
import concurrent.futures
import h5py

import ipdb

import numpy as np

from tasks import TaskPatchesCached

def extract_patches(args, task):
    X = np.zeros((args.n_examples, 3, args.patch_size, args.patch_size)).astype(np.float32)
    print('X: ', X.shape)
    if args.n_examples > task.N_crops:
        print('should extract %d, but only %d detections - some will be sampled multiple times.')

    with ProgressBar(max_value=args.n_examples) as progress:
        for idx, r in enumerate([task(x % task.N_crops) for x in range(args.n_examples)]):
            X[idx, :] = r[0] # ignore labels - r[1]
            progress.update(idx)
    return X

def compute_stats(args, X):
    if args.pixelwise:
        mean = np.mean(X, axis=(0,2,3), dtype=np.float32)
        std = np.std(X, axis=(0,2,3), dtype=np.float32)
        print('float32: ', mean, std)

        mean = np.mean(X.astype(np.float64), axis=(0,2,3), dtype=np.float64)
        std = np.std(X.astype(np.float64), axis=(0,2,3), dtype=np.float64)
        print('float64: ', mean, std)
    else:
        mean = np.mean(X.astype(np.float64), axis=0, dtype=np.float64)
        std = np.std(X.astype(np.float64), axis=0, dtype=np.float64)

    return mean, std

def main(args):
    task = TaskPatchesCached(filename_crops=args.input,
                                patch_size=args.patch_size,
                                layout=np.array([3, 6]),
                                jitter=4,
                                drop_colorchannel=False)
    X = extract_patches(args, task)

    print('extracted all crops. Computing mean and std.dev...')
    mean, std = compute_stats(args, X)

    print('done. writing to file: ', args.output)
    h = h5py.File(args.output, 'w')
    h['mean'] = mean
    h['std'] = std
    h.close()
    print('success.')

    if args.verify:
        print('verifying...')
        task_with_normalize = TaskPatchesCached(filename_crops=args.input,
                                patch_size=args.patch_size,
                                layout=np.array([3, 6]),
                                jitter=4,
                                drop_colorchannel=False,
                                patch_mean = mean,
                                patch_std = std)

        XX = extract_patches(args, task_with_normalize)

        vmean, vstd = compute_stats(args, XX)

        print('verification mean: ', vmean)
        print('verification std : ', vstd)

        ipdb.set_trace()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', 
                        help='hdf5 input of detection bounding box crops')
    parser.add_argument('--n_examples', '-n', type=int, default=50000)
    parser.add_argument('--patch_size', '-p', type=int, default=32)
    parser.add_argument('--pixelwise', action='store_true', help=' if true only RGB mean&var values are computed')
    parser.add_argument('--verify', action='store_true', help='extract another dataset and see if it has mean 0 and std 1')
    parser.add_argument('output', help='hdf5 output containing mean and variance')
    args = parser.parse_args()
    main(args)
