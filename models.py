"""..."""

from __future__ import print_function
import lasagne
# import theano
# import theano.tensor as T
# import time
import numpy as np
import numpy as np
import pickle
# from load_train_data import load_dataset
from config import *


class Models:
    """.."""

    def __init__(self):
        """.."""
        pass

    def model2conv(self, input_width, input_height, output_dim,
                   batch_size=BATCH_SIZE):
        """.."""
        layer_in = lasagne.layers.InputLayer(
            shape=(None, 3, input_width, input_height))

        layer_conv1 = lasagne.layers.Conv2DLayer(
            layer_in,
            num_filters=64,
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        layer_pool1 = lasagne.layers.MaxPool2DLayer(
            layer_conv1, pool_size=(2, 2))

        layer_conv2 = lasagne.layers.Conv2DLayer(
            layer_pool1,
            num_filters=128,
            filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_pool2 = lasagne.layers.MaxPool2DLayer(
            layer_conv2, pool_size=(2, 2))

        layer_hidden1 = lasagne.layers.DenseLayer(
            layer_pool2,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_hidden1_dropout = lasagne.layers.DropoutLayer(
            layer_hidden1, p=0.5)

        layer_out = lasagne.layers.DenseLayer(
            layer_hidden1_dropout,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform())

        return layer_out

    def AlexNet_model(self,input_width,input_height, output_dim, batch_size=BATCH_SIZE):
        """.."""
        layer_in = lasagne.layers.InputLayer(
            shape=(None, 3, input_width, input_height))

        layer_conv1 = lasagne.layers.Conv2DLayer(
            layer_in,
            num_filters=48,
            filter_size=(5, 5),
            stride=2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        layer_pool1 = lasagne.layers.MaxPool2DLayer(
            layer_conv1, pool_size=(3, 3),stride=2)

        layer_conv2 = lasagne.layers.Conv2DLayer(
            layer_pool1,
            num_filters=128,
            filter_size=(3, 3),
            stride=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_pool2 = lasagne.layers.MaxPool2DLayer(
            layer_conv2, pool_size=(2, 2),stride=2)
        
        layer_conv3 = lasagne.layers.Conv2DLayer(
            layer_pool2,
            num_filters=192,
            filter_size=(3, 3),
            stride=1,
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_conv4 = lasagne.layers.Conv2DLayer(
            layer_conv3,
            num_filters=192,
            filter_size=(3, 3),
            stride=1,
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_conv5 = lasagne.layers.Conv2DLayer(
            layer_conv4,
            num_filters=128,
            filter_size=(3, 3),
            stride=1,
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


        layer_pool3 = lasagne.layers.MaxPool2DLayer(
            layer_conv5, pool_size=(2, 2),stride=2)

        layer_hidden1 = lasagne.layers.DenseLayer(
            layer_pool3,
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_hidden1_dropout = lasagne.layers.DropoutLayer(
            layer_hidden1, p=0.5)

        layer_hidden2 = lasagne.layers.DenseLayer(
            layer_hidden1_dropout,
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        layer_hidden2_dropout = lasagne.layers.DropoutLayer(
            layer_hidden2, p=0.5)


        layer_out = lasagne.layers.DenseLayer(
            layer_hidden2_dropout,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform())

        return layer_out

