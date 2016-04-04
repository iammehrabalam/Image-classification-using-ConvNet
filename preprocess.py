""".."""

from __future__ import print_function
import lasagne
import theano
import theano.tensor as T
import numpy as np
import pickle
import random


class Classification:
    """.."""

    def __init__(self):
        """.."""
        pass


class DatasetLoadAndClean:
    """.."""

    def loader_divde(self, filename, no_of_class=13, per_train=80,
                     per_validation=10, per_test=10):
        """.."""
        op = open(filename, "rb")
        data, labels = pickle.load(op)
        op.close()
        training_set = training_label = []
        validation_set = validation_label = []
        test_set = test_label = []

        k = 0
        for i in xrange(13):
            j = i * 2500
            k = (i + 1) * 2500

            training_set = training_set + data[j: j + 2000]
            training_label = training_label + labels[j: j + 2000]
            validation_set = validation_set + data[j + 2000: j + 2250]
            validation_label = validation_label + labels[j + 2000: j + 2250]
            test_set = test_set + data[j + 2250: k]
            test_label = test_label + labels[j + 2250: k]

        print(len(test_set), len(validation_set), len(training_set))
        print(len(test_label), len(validation_label), len(training_label))
        i = 0

        data_loads = zip(training_set, training_label)
        random.shuffle(data_loads)
        training_set, training_label = zip(*data_loads)

        data_loads = zip(validation_set, validation_label)
        random.shuffle(data_loads)
        validation_set, validation_label = zip(*data_loads)

        data_loads = zip(test_set, test_label)
        random.shuffle(data_loads)
        test_set, test_label = zip(*data_loads)

        training_set, training_label = np.array(training_set) / 255, np.array(training_label)
        validation_set, validation_label = np.array(validation_set) / 255, np.array(validation_label)
        test_set, test_label = np.array(test_set) / 255, np.array(test_label)

        print(training_set.shape[0])
        training_set = training_set.reshape((training_set.shape[0], 3, 32, 32))
        validation_set = validation_set.reshape((validation_set.shape[0], 3,
                                                 32, 32))
        test_set = test_set.reshape((test_set.shape[0], 3, 32, 32))

        print('Train data shape: ', training_set.shape)
        print('Train labels shape: ', training_label.shape)
        print('Validation data shape: ', validation_set.shape)
        print('Validation labels shape: ', validation_label.shape)
        print('Test data shape: ', test_set.shape)
        print('Test labels shape: ', test_label.shape)
        print(test_label)

        return dict(
            X_train=theano.shared(lasagne.utils.floatX(training_set)),
            y_train=T.cast(theano.shared(training_label), 'int32'),
            X_valid=theano.shared(lasagne.utils.floatX(validation_set)),
            y_valid=T.cast(theano.shared(validation_label), 'int32'),
            X_test=theano.shared(lasagne.utils.floatX(test_set)),
            y_test=T.cast(theano.shared(test_label), 'int32'),
            num_examples_train=training_set.shape[0],
            num_examples_valid=validation_set.shape[0],
            num_examples_test=test_set.shape[0],
            input_height=training_set.shape[2],
            input_width=training_set.shape[3],
            output_dim=13,
            ytest=test_label)

    def normalize(self):
        """.."""
        pass

    def _randomize(self):
        """.."""
        pass
