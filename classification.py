#!/usr/bin/env python

"""Example which shows with the MNIST dataset how Lasagne can be used."""

from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys

import numpy as np
import lasagne
import theano
import theano.tensor as T
from config import *


def create_iter_functions(dataset, output_layer,x,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one.
       epoch.
    """

    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    batch_slice_test = slice(batch_index * 1, (batch_index + 1) * 1)

    # objective = lasagne.objectives.Objective(
    # output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

    output_train = lasagne.layers.get_output(output_layer, X_batch)
    loss_train = lasagne.objectives.aggregate(
        lasagne.objectives.categorical_crossentropy(output_train, y_batch))

    output_eval = lasagne.layers.get_output(
        output_layer, X_batch, deterministic=True)
    loss_eval = lasagne.objectives.aggregate(
        lasagne.objectives.categorical_crossentropy(output_eval, y_batch))

    # loss_train = objective.get_loss(X_batch, target=y_batch)
    # loss_eval = objective.get_loss(X_batch, target=y_batch,
    #                                deterministic=True)

    #in lasagne old version 0.1
    # pred = T.argmax(
    #     output_layer.get_output_for(X_batch, deterministic=True), axis=1)
    
    predict = T.argmax(
        lasagne.layers.get_output(output_layer,X_batch, deterministic=True), axis=1)

    accuracy = T.mean(T.eq(predict, y_batch), dtype=theano.config.floatX)
    # pred = list of predicted indices for given inputs in batch size

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    #yo = x[5].get_output_for(X_batch, deterministic=True)
    #output of fully connected layer
    feature_matrix = lasagne.layers.get_output(x[5],X_batch, deterministic=True)
    


    iter_train = theano.function(
        [batch_index], [loss_train,predict],

        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )
    
    fully_connected_output = theano.function(
        [batch_index], [loss_train,feature_matrix],
        givens={
            X_batch: dataset['X_train'][batch_slice_test],
            y_batch: dataset['y_train'][batch_slice_test],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [accuracy, predict, feature_matrix],

        givens={
            X_batch: dataset['X_test'][batch_slice_test],
            y_batch: dataset['y_test'][batch_slice_test],
        },
    )
    #error???
    # fully_connected_output = theano.function(
    #     [batch_index], [predict,feature_matrix],
    #     givens={
    #         X_batch: dataset['X_train'][batch_slice_test],
    #         y_batch: dataset['y_train'][batch_slice_test],
    #     },
    # )
    # # iter_funcs['train'](0)

    

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        fclo=fully_connected_output,
    )




def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training.
       Each mini-batch has `batch_size` recordings.
    """

    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    print(num_batches_train, num_batches_valid)

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            # print(vars(iter_funcs['train']))
            print (b)
            # print(iter_funcs['train'](b))
            batch_train_loss , pred= iter_funcs['train'](b)

            batch_train_losses.append(batch_train_loss)
            # print (b,predi)
        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            # print (b)
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            # 'test':accu,
        }


def test(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """.."""
    y = dataset['ytest']
    num_batches_test = dataset['num_examples_test']
    print (num_batches_test)

    batch_test_accuracies = []
    prediction = []

    for i in range(1):
        batch_test_accuracy, predi , out = iter_funcs['test'](i)
        print (y[i], predi)

    for b in range(num_batches_test):
        batch_test_accuracy, predi , out  = iter_funcs['test'](b)

        batch_test_accuracies.append(batch_test_accuracy)
        prediction.append((int(y[b]), predi))
        # print(y[b],predi)
    accu = np.mean(batch_test_accuracies)
    # print(prediction[0])
    print("Test accuracy : " + str(accu * 100))

    f = open("result.txt", "w")
    f.write(str(prediction))
    f.flush()

