from __future__ import print_function
from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T
import pickle

import lasagne
from lasagne.layers import InputLayer, DenseLayer, MaxPool2DLayer, FlattenLayer, DropoutLayer
from lasagne.layers.shape import ReshapeLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvDNN
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import rectify as ReLU
from lasagne.init import Constant

from nets import make_init_function

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class ModelBaseNet(object):
    '''
    This is a model that contains a 'base net' -- to which outputs can be added.
    - provides routines to create the theano expressions for train and eval
    - provides functionality to save and load parameters
    - function to compute statistic updates
    '''
    def __init__(self, batch_size, net_input_dims, net_generator, weightfile, init_method=None):
        self.batch_size = batch_size
        self.net_input_dims = net_input_dims

        self.X = T.tensor4('X')

        self.output_names = []
        self.output_max_values = []
        self.output_gt_indices = []
        self.outputs = [] # output layers of self.net
        self.losses = [] # theano expression for the loss

        self.net_generator = net_generator

        # setup the network
        if net_generator is None:
            raise ValueError('we need a valid net generator!')
        else:
            self.net = net_generator(self.net_input_dims, self.X, initializer=init_method)

        # initialize network from weight file
        self.last_base_layer = self.net.values()[-1]
        if weightfile is not None:
            print('loading network weights from file...')
            # the file needs to be a pickle with either (2 supported formats):
            # - a dict containing a list of np.ndarrays in key 'param values' (downloaded weight files)
            # - a list of CudaNdarraySharedVariable objects with the parameters (stored lasagne params)
            data = pickle.load(open(weightfile))
            if type(data) is dict:
                weights = data['param values'] ## pretrained vgg16.pkl use this convention
                print('--> the weights look like pretrained weights from somewhere else... SCALE correct ???')
            if type(data) is list:
                # if this is a model that we have pickled using the lasagne code, we 
                # need to call get_value to extract the actual parameters
                weights = [x.get_value() for x in data]
            # we need to exclude them here (the layers have different shapes)
            #lasagne.layers.set_all_param_values(self.net['pool5'], [x.get_value() for x in weights[:-4]])
            param_count = len(lasagne.layers.get_all_params(self.last_base_layer))
            if param_count > len(weights):
                raise ValueError('this network has more parameters to set than the weightfile.')
            lasagne.layers.set_all_param_values(self.last_base_layer, weights[:param_count])
        else:
            print('no pretrained weights are given -- initializing network _RANDOMLY_')

    @property
    def loss(self):
        ''' return theano expression for loss
        '''
        # maybe make this a weighted sum later on
        # mean across batch for each loss individually - then sum over all losses
        if len(self.losses) == 0:
            raise TypeError('self.losses appears to be empty')
        elif len(self.losses) == 1:
            return T.mean(self.losses[0])
        else:
            return T.sum([T.mean(x) for x in self.losses])

    @property
    def predictions(self):
        return lasagne.layers.get_output(self.outputs, deterministic=False)

    def build_fn_train(self, loss, grads, updates, with_grad_mag=False):
        ''' the loss is self.loss + potentially some regularization etc. 
        '''
#        if isinstance(updates, OrderedDict):
#            ups = [u for p, u in updates.values()]
#        elif isinstance(updates, list):
#            ups = [u for p, u in updates]
#       FIXME properly compute the gradient norms here -- for visualization & debugging
        outputs = [loss,]

        if with_grad_mag:
            grad_mag = T.sum([ T.sqrt(T.sum(x**2)) for x in grads])
            outputs.append(grad_mag)

        self.fn_train = th.function(inputs=[self.X, self.Y],
                                    outputs=outputs,
                                    updates=updates)

    def evaluate(self, crops, targets):
        ''' fn_predict should be initialized by build_fn_predict (or manually)
        '''
        if self.fn_predict is None:
            raise Exception('fn_predict not initialized!')
        else:
            return self.fn_predict(crops, targets)

    def dump_parameters_to_file(self, filename):
        ''' dump all model parameters (not the class attributes) to a pickle file
        '''
        f = open(filename, 'w')
        pickle.dump(lasagne.layers.get_all_params(self.outputs), f)

    def restore_from_file(self, filename):
        f = open(filename)
        data = pickle.load(f)

        weights = [x.get_value() for x in data]
        params = lasagne.layers.get_all_params(self.outputs)
        if len(params) != len(weights):
            raise ValueError('parameters in weight file does not match parameters in model')
        lasagne.layers.set_all_param_values(self.outputs, weights)

    def dump_weight_stats(self, file):
        ''' takes a file object
            -> writes overview of the networks weights to this file
        '''
        if isinstance(file, str):
            try:
                file = open(file,'w')
            except IOError:
                raise ValueError('could not open weight stats file!')

        file.write('\n === network weights ===\n')
        values = lasagne.layers.get_all_param_values(self.outputs)

        file.write('{}:\t {:>18} {:>18} {:>18} {:>18} {:>18} {:>25}\n'.format('idx', 'max', 'min',
                                                                              'vmax-vmin', 'mean', 'std', 'shape'))
        for idx, val in enumerate(values):
            vmax = np.max(val)
            vmin = np.min(val)
            file.write('{}:\t {:18.10} {:18.10f} {:18.10f} {:18.10f} {:18.10f} {:>25}\n'.format(idx,
                                                                                     vmax, vmin, vmax-vmin,
                                                                                     np.mean(val), np.std(val),
                                                                                     str(val.shape)))
        file.flush()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class ModelAttributes(ModelBaseNet):
    '''
    '''
    def __init__(self, batch_size, image_input_size, net_generator, weightfile, output_names,
                 max_values, ignore_na_label = True, init_method=None, deterministic_evaluation=True,
                 restore_all_weights=False):
        self.Y = T.imatrix('Y') # input vars need to be present before the base constructor is called

        net_input_dims = (batch_size, 3, image_input_size[0], image_input_size[1])

        if restore_all_weights:
            if not weightfile:
                raise ValueError('weightfile not specified, but restore_all_weights needs it!')
            super(ModelAttributes, self).__init__(batch_size, net_input_dims, net_generator, None, init_method)
        else:
            super(ModelAttributes, self).__init__(batch_size, net_input_dims, net_generator, weightfile, init_method)
        # ---
        # init the class specific attributes by default
        self.ignore_na_label = ignore_na_label

        # ---
        # add the class specific outputs, loss, and theano functions
        print('adding output to the network...')
        print('\t\tadding flatten layer and shared FC...')
        self.net['flat'] = FlattenLayer(self.last_base_layer)

        self.net['fc_shared'] = DenseLayer(self.net['flat'],
                                           num_units=512,
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=make_init_function(init_method))

        # in case this is a batch normalized network -> also make this layer batch normalized
        layer_types = [type(x) for x in lasagne.layers.get_all_layers(self.net['flat'])]
        if lasagne.layers.normalization.BatchNormLayer in layer_types:
            self.net['fc_shared'] = batch_norm(self.net['fc_shared'])
            print('fc_shared batch normalized')

        #if lasagne.layers.DropoutLayer in layer_types:
        #    self.net['fc_shared'] = DropoutLayer(self.net['fc_shared'], p=0.5, rescale=True)
        #    print('added DropOut after fc_shared')

        self.last_common_layer = self.net['fc_shared']
        print('\t\tadding outputs...')
        print('\t\toutputs: ', output_names)
        print('\t\tvalid_labels: ', max_values)
        idx = 0
        for output_name, max_n in zip(output_names, max_values):
            print('\t\t\t', output_name, max_n)
            self._make_output(output_name, idx, max_n, self.last_common_layer)
            idx += 1

        if restore_all_weights:
            print('\t\t restoring weights from model file: ', weightfile)
            self.restore_from_file(weightfile)

        self.build_fn_eval(deterministic_evaluation)

    def _make_output(self, output_name, index, max_n, input_layer):
        # symbolic variable for the ground truth values
        one_hot_targets = T.extra_ops.to_one_hot(self.Y[:, index], max_n) # the target value

        if self.ignore_na_label:
            max_n = max_n -1
        self.net[output_name] = DenseLayer(input_layer,
                                    num_units=max_n,
                                    W=lasagne.init.Constant(val=0.0),
                                    nonlinearity=lasagne.nonlinearities.softmax)

        self.outputs.append(self.net[output_name])
        self.output_names.append(output_name)
        self.output_max_values.append(max_n)
        self.output_gt_indices.append(index)

        # theano expressions
        output_expression = lasagne.layers.get_output(self.net[output_name], deterministic=False)

        if self.ignore_na_label:
            #targets = T.set_subtensor(one_hot_targets[:, 0], 0) # the first 'column' to 0
            #loss = lasagne.objectives.categorical_crossentropy(output_expression, targets)
            loss = lasagne.objectives.categorical_crossentropy(output_expression, one_hot_targets[:, 1:])
        else: #softmax N+1
            loss = lasagne.objectives.categorical_crossentropy(output_expression, one_hot_targets)
        self.losses.append(loss)

    def build_fn_eval(self, deterministic_evaluation=True):
        ''' theano function returns [loss (unregularized!), accuracy, all predictions]
        '''
        predictions = lasagne.layers.get_output(self.outputs, deterministic=deterministic_evaluation)
        self.fn_predict = th.function(inputs=[self.X, self.Y], outputs=[self.loss]+predictions)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class ModelPatches(ModelBaseNet):
    '''
    '''
    def __init__(self, batch_size, image_input_size, net_generator, weightfile, max_value, pairwise=False, init_method=None):
        self.Y = T.ivector('Y') # input vars need to be present before the base constructor is called

        if pairwise:
            net_input_dims = (2*batch_size, 3, image_input_size[0], image_input_size[1])
        else:
            net_input_dims = (batch_size, 3, image_input_size[0], image_input_size[1])

        super(ModelPatches, self).__init__(batch_size, net_input_dims, net_generator, weightfile, init_method)

        # ---
        # init the class specific attributes

        # ---
        # add the class specific outputs, loss, and theano functions
        if pairwise:
            if self.batch_size % 2 == 1:
                raise ValueError('batch_size for pairwise tasks _must_ be even')
            # the net input dims are (2*batch_size, .... )
            self.net['reshape'] = ReshapeLayer(self.last_base_layer, (self.batch_size, -1)) # reshape and flatten

            self.net['fc6'] = DenseLayer(self.net['reshape'],
                                         num_units=256,
                                         W=make_init_function(init_method),
                                         nonlinearity=lasagne.nonlinearities.rectify)
        else:
            self.net['fc6'] = DenseLayer(self.last_base_layer,
                                         num_units=256,
                                         W=make_init_function(init_method),
                                         nonlinearity=lasagne.nonlinearities.rectify)

        # in case this is a batch normalized network -> also make this layer batch normalized
        layer_types = [type(x) for x in lasagne.layers.get_all_layers(self.last_base_layer)]
        if lasagne.layers.normalization.BatchNormLayer in layer_types:
            self.net['fc6'] = batch_norm(self.net['fc6'])
            print('fc6 batch normalized')

        self.net['output'] = DenseLayer(self.net['fc6'],
                                   num_units=max_value,
                                   W=lasagne.init.Constant(val=0.0),
                                   nonlinearity=lasagne.nonlinearities.softmax)
        self.outputs.append(self.net['output'])
        self.output_names.append('patch_class')
        self.output_max_values.append(max_value)

        # theano expressions
        output_expression = lasagne.layers.get_output(self.net['output'], deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(output_expression, self.Y)
        self.losses.append(loss)

        self.build_fn_eval()

    def build_fn_eval(self):
        ''' theano function returns [loss (unregularized!), accuracy, all predictions]
        '''
        predictions = lasagne.layers.get_output(self.outputs[0], self.X, deterministic=True)
        accuracy = T.mean(T.eq(T.argmax(predictions, axis=1), self.Y))
        self.fn_predict = th.function(inputs=[self.X, self.Y], outputs=[self.loss, 100.*accuracy])
        #self.fn_predict = th.function(inputs=[self.X, self.Y],
        #                              outputs=[self.loss, 100.*accuracy]+predictions)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#class PairwisePatchModel(ModelBaseNet):
#    pass
#
