#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import h5py
from matplotlib import pyplot as plt
from argparse import ArgumentParser

def read_patches(args):
    if args.verbose:
        print('reading from file: ', args.input_file)
    if args.verbose:
        print('mode: hdf5')
    h = h5py.File(args.input_file, 'r')
    # if present this will return the pids - otherwise None
    return h['patches'], h['labels']

def main(args):
    print('loading patches from: ', args.input_file)
    patches, labels = read_patches(args)
    print('patches: ', patches.shape)

    indices = range(len(patches))
    if args.index:
        if args.index < 0 or args.index >= len(patches):
            print('invalid index - valid are 0..', len(patches))
            sys.exit(1)
        indices = [args.index]
    if args.random:
        np.random.shuffle(indices)
    for iii in indices:
        print('idx: ', iii, 'label: ', labels[iii])
        c = patches[iii, :]
        t = c.transpose((2,1,0))
        plt.imshow(t)

        title = 'example idx: ' + str(iii) + ' label: ' + str(labels[iii])
        plt.title(title)
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--random', '-r', action='store_true', help='random order')
    parser.add_argument('--index', '-i', type=int, help='only visualize this index')
    parser.add_argument('input_file')
    args = parser.parse_args()

    main(args)
