from __future__ import print_function
from collections import OrderedDict

import lasagne
from lasagne.layers import InputLayer, DenseLayer, MaxPool2DLayer, FlattenLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvDNN
from lasagne.layers.noise import dropout
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import rectify as ReLU
from lasagne.init import HeNormal, HeUniform, Constant, Orthogonal, GlorotNormal, GlorotUniform

def make_init_function(init_method, gain='relu'):
    if init_method == 'he_normal' or init_method is None:
        ini = HeNormal(gain=gain)
    elif init_method == 'he_uniform':
        ini = HeUniform(gain=gain)
    elif init_method == 'orthogonal':
        ini = Orthogonal(gain=gain)
    elif init_method == 'glorot_normal':
        ini = GlorotNormal(gain=gain)
    elif init_method == 'glorot_uniform':
        ini = GlorotUniform(gain=gain)
    else:
        raise ValueError('invalid init_method')
    return ini



#
# NOTE
#
# these are some example nets ... 
# others can be implemented here as methods -- and then directly be used for the script 'patch_training.py'
#


def vgg16_no_bn(net_input_dims, input_var=None, initializer=None):
    ini = make_init_function(initializer)

    net = OrderedDict()
    net['input'] = InputLayer(net_input_dims, input_var=input_var)
    net['conv1_1'] = ConvDNN(net['input'], 64, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv1_2'] = ConvDNN(net['conv1_1'], 64, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvDNN(net['pool1'], 128, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv2_2'] = ConvDNN(net['conv2_1'], 128, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvDNN(net['pool2'], 256, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv3_2'] = ConvDNN(net['conv3_1'], 256, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv3_3'] = ConvDNN(net['conv3_2'], 256, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvDNN(net['pool3'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv4_2'] = ConvDNN(net['conv4_1'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv4_3'] = ConvDNN(net['conv4_2'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvDNN(net['pool4'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv5_2'] = ConvDNN(net['conv5_1'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv5_3'] = ConvDNN(net['conv5_2'], 512, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool5'] = MaxPool2DLayer(net['conv5_3'], 2)
    return net


def tiny_10_dropout(net_input_dims, input_var=None, initializer=None):
    ini = make_init_function(initializer)

    net = OrderedDict()
    net['input'] = InputLayer(net_input_dims, input_var=input_var)

    net['conv1_1'] = ConvDNN(net['input'], 16, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv1_2'] = ConvDNN(net['conv1_1'], 16, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['drop1'] = DropoutLayer(net['pool1'], p=0.5, rescale=True)

    net['conv2_1'] = ConvDNN(net['drop1'], 32, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv2_2'] = ConvDNN(net['conv2_1'], 32, 3, pad=1, nonlinearity=ReLU, W=ini)

    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['drop2'] = DropoutLayer(net['pool2'], p=0.5, rescale=True)

    net['conv3_1'] = ConvDNN(net['drop2'], 64, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv3_2'] = ConvDNN(net['conv3_1'], 64, 3, pad=1, nonlinearity=ReLU, W=ini)

    net['pool3'] = MaxPool2DLayer(net['conv3_2'], 2)
    net['drop3'] = DropoutLayer(net['pool3'], p=0.5, rescale=True)

    net['conv4'] = ConvDNN(net['drop3'], 128, 3, pad=1, nonlinearity=ReLU, W=ini)
    net['conv4_2'] = ConvDNN(net['conv4'], 128, 3, pad=1, nonlinearity=ReLU, W=ini)

    net['pool4'] = MaxPool2DLayer(net['conv4_2'], 2)
    net['drop4'] = DropoutLayer(net['pool4'], p=0.5, rescale=True)
    return net

