""".."""

from __future__ import print_function
import lasagne
import theano
import theano.tensor as T
import numpy as np
import pickle
import random
width = height = 32

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
        data, labels, pathname= pickle.load(op)
        op.close()
        training_set = training_label = training_pathname = []
        validation_set = validation_label = validation_pathname = []
        test_set = test_label = test_pathname = []

        k = 0
        for i in xrange(13):
            j = i * 2500
            k = (i + 1) * 2500

            training_set = training_set + data[j: j + 2000]#2000
            training_label = training_label + labels[j: j + 2000]
            training_pathname = training_pathname + pathname[j: j + 2000]
            
            validation_set = validation_set + data[j + 2000: j + 2250]#2250
            validation_label = validation_label + labels[j + 2000: j + 2250]
            validation_pathname = validation_pathname + pathname[j + 2000: j + 2250]

            test_set = test_set + data[j + 2250: k]
            test_label = test_label + labels[j + 2250: k]
            test_pathname = test_pathname + pathname[j + 2250: k]


        print(len(test_set), len(validation_set), len(training_set))
        print(len(test_label), len(validation_label), len(training_label))
        print(len(test_pathname), len(validation_pathname), len(training_pathname))

        print(test_set[0].shape)
        i = 0

        data_loads = zip(training_set, training_label, training_pathname)
        random.shuffle(data_loads)
        training_set, training_label, training_pathname = zip(*data_loads)

        data_loads = zip(validation_set, validation_label, validation_pathname)
        random.shuffle(data_loads)
        validation_set, validation_label, validation_pathname= zip(*data_loads)

        data_loads = zip(test_set, test_label, test_pathname)
        random.shuffle(data_loads)
        test_set, test_label, test_pathname = zip(*data_loads)

        training_set, training_label = np.array(training_set) / 255, np.array(training_label)
        validation_set, validation_label = np.array(validation_set) / 255, np.array(validation_label)
        test_set, test_label = np.array(test_set) / 255, np.array(test_label)

        print(training_set.shape[0])
        training_set = training_set.reshape((training_set.shape[0], 3, width, height))
        validation_set = validation_set.reshape((validation_set.shape[0], 3,
                                                 width, height))
        test_set = test_set.reshape((test_set.shape[0], 3, width, height))

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
            ytest=test_label,
            ytrain=training_label,
            training_pathname=training_pathname,
            validation_pathname=validation_pathname,
            test_pathname=test_pathname)



    def normalize(self):
        """.."""
        pass



    def _randomize(self):
        """.."""
        pass

