""".."""

from __future__ import print_function
import lasagne
import theano.tensor as T
import time
import numpy as np
import cPickle as pickle
from PIL import Image
import  matplotlib.pyplot as plt 
from pylab import *

from classification import create_iter_functions
from classification import train
from classification import test
from preprocess import DatasetLoadAndClean
from config import *
from models import Models

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler('epoch.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Image_classification(object):
    
    def __init__(self):
        print("Loading data...")
        self.dataset = DatasetLoadAndClean().loader_divde("v3shoes_dataset32_label_pathnames.pickle")
        print("Building model and compiling functions...")

        self.output_layer = Models().model2conv(
            input_height=self.dataset['input_height'],
            input_width=self.dataset['input_width'],
            output_dim=self.dataset['output_dim'])

        self.x = lasagne.layers.get_all_layers(self.output_layer)

        self.iter_funcs = create_iter_functions(
            self.dataset,
            self.output_layer,
            self.x,
            X_tensor_type=T.tensor4)
    
    def training(self,num_epochs=NUM_EPOCHS):
        """.."""

        

        # with np.load('epoch-1new.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        # lasagne.layers.set_all_param_values(output_layer, param_values)

        # test(iter_funcs, self.dataset, x)
        

        print("Starting training...")
        now = time.time()
        try:
            for epoch in train(self.iter_funcs, self.dataset):
                logger.info("Epoch {} of {} took {:.3f}s".format(epoch['number'], num_epochs, time.time() - now))
                now = time.time()
                logger.info("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
                logger.info("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
                logger.info("  validation accuracy:\t\t{:.2f} %%".format(epoch['valid_accuracy'] * 100))

                if epoch['number'] >= num_epochs:
                    break

        except KeyboardInterrupt:
            pass

        st = 'epoch' + '_v3'
        np.savez(st + '.npz', *lasagne.layers.get_all_param_values(self.output_layer))

        return self.output_layer

    #if __name__ == '__main__':
    #    main()

    
    def testing(self):
        with np.load('epoch_v3.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(self.output_layer, param_values)

        test(self.iter_funcs, self.dataset, self.x)


    def extract_features(self):
        """.."""


        with np.load('epoch_v3.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.output_layer, param_values)


        num_batches_train=self.dataset["num_examples_train"]
        training_pathname=self.dataset["training_pathname"]
        training_label=self.dataset["ytrain"]

        output_dim=self.dataset['output_dim']
        features={}
        for i in xrange(output_dim):
            features[i]=[]

        print(training_pathname[:5])
        print(training_label[:5])
        print(num_batches_train)

        for i in range(num_batches_train):
            print (i)
            loss, out  = self.iter_funcs['fclo'](i)
            #print(type(out), len(out), out)
            features[training_label[i]].append((out,training_pathname[i]))

                        
        pkl = open("v3feature_vector.pickle", "wb")
        pickle.dump(features, pkl, pickle.HIGHEST_PROTOCOL)#for binary file protocol>=1
        pkl.close()        



    def visually_similar(self):
        """......"""
        with np.load('epoch_v3.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.output_layer, param_values)

        test_pathname=self.dataset["test_pathname"]
        test_label=self.dataset["ytest"]
        
        fl = open("v3feature_vector.pickle")
        feature_vector = pickle.load(fl)#from training images

        print("Enter Test Image index :")
        scan = input()
        if scan < 0 and scan >= 3000:
            print("choose index between 0 and 3000 ")
            return

        print (test_pathname[scan])    
        arr = np.array(Image.open(test_pathname[scan]))    
        #plt.show(arr)
        imshow(arr)
        title(test_pathname[scan])
        show()

        '''
        dic={}
        for i in xrange(13):
            dic[i]=[]
        try:
            for j in xrange(3500):
                loss, pred, out = self.iter_funcs['test'](j)
                if pred[0]!=test_label[j]: 
                    print (pred[0],test_label[j])
                    dic[test_label[j]].append(test_pathname[j])
        except:
            print ("a")            
        f = open("abc.txt","w")
        f.write(str(dic))
        f.close()
        return        

        '''

        loss, pred, out = self.iter_funcs['test'](scan)    

        print("prediction : " + str(pred))
        print("Ground truth label : " + str(test_label[scan]))


        features , pathname = zip(*feature_vector[pred[0]])
        print(len(features),type(pathname))

        euclidean_dist=[]
        for i in xrange(len(features)):
            #print (arr.shape,features[i].shape)
            x = features[i]  - out
            x = np.square(x)  
            euclidean_dist.append(np.sqrt(sum(x)))

        sortme = zip(euclidean_dist,pathname)
        sortme.sort()
        euclidean_dist,pathname=zip(*sortme)
        
        print (euclidean_dist[:10])
        print(test_pathname[:10],)

        for i in xrange(10):
            arr = np.array(Image.open(pathname[i])) 
            print(pathname[i])   
            imshow(arr)
            title(pathname[i])
            show()







