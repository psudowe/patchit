#!/usr/bin/env python

"""
Takes a file with detection bounding boxes.
Crops the boxes and saves all resulting crops in a .hdf5 file.
This can then be used to do pre-training tasks.
"""

from __future__ import print_function
import os, sys, time, os.path as path
from argparse import ArgumentParser
from progressbar import ProgressBar
import concurrent.futures
import h5py

import numpy as np
from skimage.util import pad
from skimage import transform
from PIL import Image

from pUtils import git, dump_args, recreate_dir

from patch_helpers import read_detections

def crop(imgfn, box, target_size, padding, padding_mode='zero', mirror=False):
    '''
    Load an image and crop bounding box from it.
    Handles additional padding - if the box is too close to the image boundary,
    the image is padded according to args.padding_mode (i.e. edge or zero)

    Parameters
    ----------
      box: the bounding box to be cropped - tuple (min_x, min_y, max_x, max_y)

      target_size: (2-tuple), scale each image s.t. the bounding box
         is of height target_size[1]
         ( then adapt in x-direction s.t. it matches target_size[0])

      padding: number of pixels of additional padding around the bounding box

      padding_mode: 'zero' or 'edge' - controls how the padded pixels are filled

      mirror: if true - the resulting crop is mirrored (reversed x-axis)

    Return
    ------
      a crop of shape: target_size + padding
      type: np.array in ordering [c,w,h] - of type uint8 for compact memory footprint
    '''
    img = np.array(Image.open(imgfn).convert('RGB'))
    cur_h, cur_w, cur_c = img.shape

    # 1. rescale the whole image s.t. bounding box has target height
    # 2. adapt box accordingly (scale to the new image dim, then adapt width)
    # 3a. add additional 'padding' to bounding box
    # 3b. add padding around the image - in case it is required
    # 4. take the crop
    # 5. transpose the dimensions
    sf = float(target_size[1]) / (box[3]-box[1])

    # 1.
    img_l = transform.resize(img, (int(cur_h * sf), int(cur_w * sf), cur_c))
    pb = [np.floor(sf*x + .5) for x in box]

    # 2.
    delta = (target_size[0]-(pb[2]-pb[0])) / 2.0
    pb[0] -= np.floor(delta + 0.5)
    pb[2] += np.floor(delta)
    if pb[2]-pb[0] <> target_size[0]:
        raise Exception('new box width does not match target')
    if pb[3]-pb[1] <> target_size[1]:
        raise Exception('new box height does not match target')

    # 3a
    pb[0] -= padding # the padding around bounding box (not the whole image)
    pb[1] -= padding
    pb[2] += padding
    pb[3] += padding
    pb = [int(x) for x in pb]

    # 3b
    if pb[0] < 0 or pb[1] < 0 or pb[2] >= img_l.shape[1] or pb[3] >= img_l.shape[0]:
        pad_offset = np.max(target_size) + padding
        pb = [int(x+pad_offset) for x in pb]

        if padding_mode == 'edge':
            img = pad(img_l, [(pad_offset, pad_offset),
                              (pad_offset, pad_offset),
                              (0,0)], mode='edge')
        elif padding_mode == 'zero':
            img = pad(img_l, [(pad_offset, pad_offset),
                              (pad_offset, pad_offset),
                              (0,0)], mode='constant', constant_values=0)
        else:
            raise NotImplemented('padding mode not implemented: ', padding_mode)
    else:
        img = img_l # no extra padding around the image required

    # 4.
    if mirror:
        acrop = img[pb[1]:pb[3], pb[2]:pb[0]:-1, :] # reversed x-dimension
    else:
        acrop = img[pb[1]:pb[3], pb[0]:pb[2], :]
    out = (255. * acrop).astype(np.uint8)
    return out.transpose((2,1,0)) # transpose to (c,w,h)

def do_one_crop(detection):
    filename, minx, miny, maxx, maxy = detection
    fullpath = os.path.join(args.sequence_prefix, filename)
    c = crop(fullpath, (minx, miny, maxx, maxy), (args.width, args.height),
             padding=0, padding_mode='zero', mirror=False)
    return c

def preprocess_detections(args, detections):
    '''
    '''
    N = len(detections)
    crops = np.zeros((N, 3, args.width, args.height), dtype=np.uint8)

    with ProgressBar(max_value=len(detections)) as progress:
        pex = concurrent.futures.ProcessPoolExecutor(max_workers=None)
        for idx, crop in enumerate(pex.map(do_one_crop, detections)):
            crops[idx, :] = crop
            if idx % 50 == 0:
                progress.update(idx)
    return crops


def main(args):
    detections = read_detections(args.input)

    if args.debug: 
        detections = detections[:500]

    if args.n_max > 0 and len(detections) > args.n_max:
        detections = detections[:args.n_max]

    crops = preprocess_detections(args, detections)
    print('extracted %d crops'%len(crops))

    output_file = os.path.join(args.output_dir, 'detection_crops.hdf5')
    print('writing to file: ', output_file)
    h = h5py.File(output_file, 'w')
    h.create_dataset('crops', data=crops)
    h.close()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--output_dir', '-o',
                        default=os.environ.get('OUTPUT_DIR', '/tmp/detection_crops'))
    parser.add_argument('--sequence_prefix', '-s',
                        default='/work/sudowe/datasets/parse_full_sequences/sequences')
    parser.add_argument('--height', type=int, default=200)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--n_max', type=int, default=-1,
                        help='maximum number of detections to process - default: process all')

    parser.add_argument('input', help='detection file')

    args = parser.parse_args()
    args.git_revision = git.current_revision(path.dirname(__file__))

    recreate_dir(args.output_dir)
    # save parameters to file (for reproducibility of results)
    dump_args(args, args.output_dir)

    main(args)
