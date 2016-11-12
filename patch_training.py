#!/usr/bin/env python

'''
Train a network using the PatchTask

The resulting weights can be used as initialization for other tasks 
on the same training data.

This code illustrates the procedure described in:
@InProceedings{Sudowe16BMVC,
    author = {Patrick Sudowe and Bastian Leibe},
    title  = {{PatchIt: Self-Supervised Network Weight Initialization for Fine-grained Recognition}},
    booktitle = BMVC,
    year = {2016}
}
'''

from __future__ import print_function
import sys, os, time
import os.path as path
from argparse import ArgumentParser
from datetime import datetime
import concurrent.futures
import h5py
import pickle

import ipdb
from parse_evaluation import evaluate_retrieval, evaluate_classification

import numpy as np
import theano as th
import theano.tensor as T

import lasagne
from lasagne.regularization import regularize_layer_params, l1, l2

from collections import namedtuple
from matplotlib import pyplot as plt

from pUtils import git, dump_args, recreate_dir

from tasks import TaskPatchesCached, TaskPairwisePatches
from patch_helpers import read_detections

from model import ModelPatches
import nets

def do_eval(data, model, report, pairwise=False):
    ts_val = time.time()

    overall_loss = 0.
    overall_acc = 0.

    N = args.batch_size*(data[1].shape[0] // args.batch_size)
    print('N: ', N, '# examples: ', data[1].shape[0])

    X_size = args.batch_size * 2 if pairwise else args.batch_size

    print('debug: X_size: ', X_size, '\t\t model net_input_dims: ', model.net_input_dims)

    for batch_idx in range(data[1].shape[0] // args.batch_size):
        batch_crops = np.zeros((X_size, 3, args.patch_size, args.patch_size), dtype=np.float32)

        idx = range(batch_idx*args.batch_size, (batch_idx+1)*args.batch_size)
        for within_batch_idx, dataset_idx in enumerate(idx):
            if pairwise:
                batch_crops[within_batch_idx*2, :] = data[0][dataset_idx*2, :]
                batch_crops[within_batch_idx*2+1, :] = data[0][dataset_idx*2+1, :]
            else:
                batch_crops[within_batch_idx,:] = data[0][dataset_idx,:]

        outputs = model.evaluate(batch_crops, data[1][idx])
        loss = outputs[0]
        accuracy = outputs[1]
        # outputs[2:] are the actual predictions, which we ignore for now
        overall_loss += loss
        overall_acc += accuracy

    print('==== patch validation set =====')
    batch_count = (data[0].shape[0] // args.batch_size)
    print('  mean loss: ', overall_loss / batch_count)
    print('  mean acc: ', overall_acc / batch_count )
    print('=== took: %d seconds ==='%(time.time() - ts_val,))
    report.write('==== patch validation set =====\n')
    report.write('  mean loss: %f\n'%(overall_loss / batch_count,))
    report.write('  mean acc: %f\n\n'%(overall_acc / batch_count,))
    report.flush()
    return (overall_acc / batch_count)

def main(args):
    print('read mean&std from file...')
    h = h5py.File(args.mean_file)
    patch_mean = h['mean'][:].astype(np.float32) # this is in [0,1] format
    patch_std  = h['std'][:].astype(np.float32)

    if args.no_std_scaling:
        print('will only shift not standard scale the examples')
        patch_std = None


    # check that patch_jitter has valid value
    if args.patch_task == 'pt18':
        if args.patch_jitter > 4:
            raise ValueError('pt18 cannot be used with patch_jitter > 4!')
        if args.patch_margin > 0:
            raise ValueError('pt18 cannot be used with patch margin > 0!')

    if args.patch_task == 'pt8':
        if args.patch_jitter > (args.patch_margin + args.patch_size//2 - 1):
         raise ValueError('pt18 cannot be used with patch_jitter > 15!')

    print('initialize task object...')
    if args.pairwise:
        task = TaskPairwisePatches(filename_crops=args.detection_crops,
                                   patch_size=args.patch_size,
                                   layout=np.array([3, 6]),
                                   jitter=args.patch_jitter,
                                   drop_colorchannel=args.drop_color_channel,
                                   patch_mean=patch_mean,
                                   patch_std=patch_std,
                                   range_to_0_255=args.input_range_0_255,
                                   patch_margin=args.patch_margin)
    else:
        if args.patch_task == 'pt18':
            pt_layout = np.array([3, 6])
        else:
            pt_layout = np.array([2, 4])
        task = TaskPatchesCached(filename_crops=args.detection_crops,
                                 patch_size=args.patch_size,
                                 layout=pt_layout,
                                 jitter=args.patch_jitter,
                                 drop_colorchannel=args.drop_color_channel,
                                 patch_mean=patch_mean,
                                 patch_std=patch_std,
                                 range_to_0_255=args.input_range_0_255,
                                 patch_margin=args.patch_margin)
    N_crops = task.crops.shape[0]

    base_net_generator = getattr(nets, args.model_type, None)
    if base_net_generator is None or not hasattr(base_net_generator, '__call__'):
        raise ValueError('invalid choice of model type (base net)')

    model = ModelPatches(batch_size=args.batch_size,
                         image_input_size=(args.patch_size, args.patch_size),
                         net_generator=base_net_generator,
                         weightfile=args.pretrained_weights,
                         max_value=task.label_max,
                         pairwise=(task.patches_per_call == 2),
                         init_method=args.init)

    if args.regularize:
        print('applying regularization...')
        if args.reg_type == 'l2':
            regularizer = l2
        elif args.reg_type == 'l1':
            regularizer = l1
        else:
            raise ValueError('invalid value for reg_type')
        reg = regularize_layer_params(model.outputs[0],  regularizer)
        loss = model.loss + args.weight_regularization * reg
    else:
        print('not applying any regularization...')

    # update expressions
    print('\t update expressions')
    learning_rate = th.shared(name='learning_rate', value=np.array(args.learning_rate, dtype=th.config.floatX))
    params = lasagne.layers.get_all_params(model.outputs, trainable=True)
    grads = th.grad(loss, params)
    if args.optimizer == 'SGD_nesterov':
        print('optimizer: SGD with Nesterov momentum')
        updates = lasagne.updates.nesterov_momentum(grads, params,
                                                    learning_rate=learning_rate, momentum=0.9)
    elif args.optimizer == 'RMSprop':
        print('optimizer: RMSprob')
        print('learning rate: ', learning_rate.get_value())
        updates = lasagne.updates.rmsprop(grads, params, learning_rate=learning_rate)
    elif args.optimizer.lower() == 'adam':
        print('optimizer: ADAM')
        print('learning rate: ', learning_rate.get_value())
        updates = lasagne.updates.adam(grads, params,
                                          learning_rate=learning_rate,
                                          beta1=0.9,
                                          beta2=0.999,
                                          epsilon=1e-08)
    else:
        raise ValueError('unknown optimizer value' + args.optimizer)

    model.build_fn_train(loss=loss, grads=grads, updates=updates, with_grad_mag=args.with_update_stats)

    #if args.verbose:
    #    #graphfile = path.join(args.output_dir, "pydotprint_trainfn__%s.png" % args.timestamp)
    #    #print('saving train graph to: ', graphfile)
    #    #th.printing.pydotprint(model.fn_train, outfile=graphfile, var_with_name_simple=True)
    #    f = path.join(args.output_dir, "debug_graph__fn_train")
    #    th.printing.debugprint(model.fn_train, file=open(f, 'w'), print_type=True)

    print('read test data...')
    h = h5py.File(args.testdata, 'r')
    test_crops = h['patches'][:].astype(np.float32)
    test_labels = h['labels'][:].astype(np.int32)
    # verify that the test set actually is as we expect
    if np.any(test_crops > 5.):
        raise ValueError('test set probably not normalized correctly')
    if np.count_nonzero(test_crops < 0.) == 0:
        raise ValueError('test set probably not normalized correctly')

    report = open(path.join(args.output_dir, 'report.log'), 'w')
    report.write(' patch task training ...\n\n')
    report.write('args: '+ str(vars(args)) +'\n\n')
    report.write( str(vars(task)) + '\n\n') 

    if args.with_update_stats:
        stats_file = open(path.join(args.output_dir, 'update_stats.txt'), 'a')
    else:
        stats_file = None

    if np.min(test_labels) != 0 or np.max(test_labels) != task.label_max-1:
        raise ValueError('something is fishy!!! this task expects labels in [0..%d]!'%(task.label_max-1,))

    if not args.quick:
        model_weight_stats_file = path.join(args.output_dir, 'model_weight_stats_init.txt')
        model.dump_weight_stats(model_weight_stats_file)
        print('\ninitial model weight stats written to: ', model_weight_stats_file)

    print('start training...')
    best_acc = -1. # 

    for epoch_idx in range(args.max_epochs):
        if args.learning_rate_decay_step > 0:
            if epoch_idx % args.learning_rate_decay_step == 0 and epoch_idx > 0:
                # time to reduce the learning rate again
                old_lr = learning_rate.get_value()
                new_lr = np.array(args.learning_rate_decay_factor, dtype=np.float32) * old_lr
                learning_rate.set_value(new_lr)
                print('\n --- learning rate decay -- %f --> %f\n'%(old_lr, new_lr))

        print('starting epoch: ', epoch_idx)
        report.write('starting epoch: ' + str(epoch_idx) +'\n')
        ts_start = time.time()
        total_epoch_loss = 0.

        n_batches = N_crops // args.batch_size
        ordering = range(N_crops)
        np.random.shuffle(ordering)

        for batch_idx in range(n_batches):
            indices = ordering[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            X = np.zeros((task.patches_per_call * args.batch_size, 3, args.patch_size, args.patch_size)).astype(np.float32)
            Y = np.zeros(args.batch_size, dtype=np.int32)
            for iii, r in enumerate([task(x) for x in indices]): # no concurrency
                if isinstance(r[0], tuple):
                    # task returns tuple of patches
                    if len(r[0]) == 2:
                        X[2*iii, :] = r[0][0]
                        X[2*iii+1, :] = r[0][1]
                    else:
                        raise ValueError('we currently can only handle pairwise patch tasks...')
                else:
                    X[iii, :] = r[0]
                Y[iii] = r[1]

            out = model.fn_train(X, Y)
            if len(out) == 2:
                batch_loss, grad_mag = model.fn_train(X, Y)
            else:
                batch_loss = out[0]
                grad_mag = None

            total_epoch_loss += batch_loss
            if grad_mag is None:
                print('\r%#6d / %d\t%.8f\t\t mean: %.8f'%(batch_idx, n_batches,
                                                        batch_loss, total_epoch_loss/(batch_idx+1)), end='')
                report.write('%#6d / %d\t%.8f\t\t mean: %.8f\n' % (batch_idx, n_batches,
                             batch_loss, total_epoch_loss/(batch_idx+1)))
            else:
                print('\r%#6d / %d\t%.8f\t\t mean: %.8f\t gradmag: %.8f'%(batch_idx, n_batches,
                                                        batch_loss, total_epoch_loss/(batch_idx+1),
                                                        grad_mag), end='')
                report.write('%#6d / %d\t%.8f\t\t mean: %.8f\t gradmag: %.8f\n' % (batch_idx, n_batches,
                             batch_loss, total_epoch_loss/(batch_idx+1),
                             grad_mag))
            report.flush()
            sys.stdout.flush()

            if stats_file is not None:
                if grad_mag is None:
                    stats_file.write('%.8f\n'%(batch_loss,))
                else:
                    stats_file.write('%.8f\t%.8f\n'%(batch_loss, grad_mag))
                stats_file.flush()

        print('\n=== epoch took: %d seconds ===\n'%(time.time() - ts_start,))
        report.write('\n=== epoch took: %d seconds ===\n'%(time.time() - ts_start,))

        model_weight_stats_file = path.join(args.output_dir, 'model_weight_stats_%d.txt'%epoch_idx)
        model.dump_weight_stats(model_weight_stats_file)
        print('\nmodel weight stats written to: ', model_weight_stats_file)
        report.write('\nmodel weight stats written to: %s\n'%model_weight_stats_file)

        # val
        print('\n == evaluation == \n')
        report.write('\n == evaluation == \n')
        report.flush()
        cur_acc = do_eval((test_crops, test_labels), model, report, args.pairwise)
        
        if (epoch_idx % 20) == 0 or cur_acc > best_acc or not args.quick:
            if cur_acc > best_acc:
                print('\n  ====> new best model: ', cur_acc, '\n')
            best_acc = cur_acc
            filename_snapshot = path.join(args.output_dir, 'patch_train_state_%s_epoch_%d'%(args.timestamp, epoch_idx,))
            model.dump_parameters_to_file(filename_snapshot)
            print('model parameters saved to: ', filename_snapshot)

    report.close()
    if stats_file is not None:
        stats_file.close()


if __name__ == '__main__':
    output_dir_prefix = os.getenv('OUTPUT_DIR_PREFIX', '/fast_work/sudowe/')
    timestamp = datetime.now().strftime('%Y_%m_%d__%H:%M:%S')
    pid = os.getpid()

    choices_model_type = [k for k,v in nets.__dict__.items() if hasattr(v, '__call__')]

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--with_update_stats', action='store_true', help='keep log of every update (slow)')

    parser.add_argument('--quick', action='store_true')

    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--max_epochs', '-m', type=int, default=20)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.5,
                        help='learning rate step decay factor')
    parser.add_argument('--learning_rate_decay_step', type=int, default=-1,
                        help='how many epochs to do before the next LR decay step (default: -1 -> no decay)')

    parser.add_argument('--optimizer', choices=['SGD_nesterov', 'RMSprop', 'ADAM'],
                        default='SGD_nesterov', help='optimization procedure')

    parser.add_argument('--regularize', '-r', action='store_true',
                        help='enable regularization (e.g. L2) of the loss')
    parser.add_argument('--weight_regularization', '-w', type=float, default=0.001,
                        help='l2 weight decay (full network) -- note this has no effect if -r is not set!')
    parser.add_argument('--reg_type', choices=['l1', 'l2'], default='l2', help='type of regularization')


    parser.add_argument('--patch_size', '-p', type=int, default=32)

    parser.add_argument('--patch_task', default='pt18', choices=('pt18', 'pt8'))
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--patch_jitter', default=4, type=int, help='#pixels to jitter')
    parser.add_argument('--patch_margin', default=0, type=int, help='#pixels margin between patches')

    parser.add_argument('--no-std-scaling', action='store_true',
                        help='only apply shift to training examples, but not scaling by standard deviation')
    parser.add_argument('--input-range-0-255', action='store_true', help='scale network input to [0,255] from [0,1]')
    parser.add_argument('--drop-color-channel', action='store_true',
                        help='randomly drop one color channel completely - default used to be True - now, default is False')


    parser.add_argument('--output_dir', '-o',
                        default=path.join(output_dir_prefix, 'patch_task__%s_%d'%(timestamp,pid)),
                        help='directory for results (models and reports)')

    parser.add_argument('--pretrained_weights', default=None, help='pretrained weight file -- if applicable')
    parser.add_argument('--init', default='he_normal', choices=['he_normal','glorot_uniform'],
                        help='random initialization of weights not initialized by optional pretrained weights')



    parser.add_argument('--model_type', choices=choices_model_type,
                        default='vgg16_with_bn', help='the underlying network to use (base net)')
    parser.add_argument('mean_file', help='hdf5 file with mean and std.deviation of training patches')
    #'/fast_work/sudowe/detection_crops/detection_crops.hdf5'
    parser.add_argument('detection_crops', help='file of detection crops to use in the patch task')
    parser.add_argument('testdata', help='hdf5 of preprocessed test data')

    args = parser.parse_args()
    args.timestamp = timestamp
    args.hostname = os.uname()[1]
    args.git_revision = git.current_revision(path.dirname(__file__))

    recreate_dir(args.output_dir)
    print('my log directory: ', args.output_dir)

    dump_args(args, args.output_dir)
    main(args)
