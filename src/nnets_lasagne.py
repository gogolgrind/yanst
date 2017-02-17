import lasagne
from lasagne import regularization


import theano
import time
import theano.tensor as T

from utils import Utils as utils
import pickle
import os
import numpy as np
import seaborn as sns
from models import Models as models

'''
refs and links:
    https://arxiv.org/pdf/1601.05610v1.pdf
    https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
'''

class Nnet():
    
    def __init__(self,imshape = (32,64),channel = 1,num_epochs = 1000,
                     lambda1 = 1e-3,lambda2 = 1e-3, learning_rate = 0.001,batch_size=512):
        self.net = {}
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.IMAGE_H = imshape[0]
        self.IMAGE_W = imshape[1]
        self.channel = channel
        self.model_name = 'roadar-net'
        self.net = self.build_model()
        self.num_epochs = num_epochs
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.t_loss = []
        self.v_loss = []        
        print('building loss ...')
        self.train_fn,self.val_fn,self.pred_fn = self.build_loss()

    def build_model(self):
        return models.build_roadar_full(shape = (self.channel,self.IMAGE_H,self.IMAGE_W), input_var = self.input_var)
        
    def iterate_minibatches(self,inputs, targets, batchsize = 128, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            inputs, targets = utils.shuffle(inputs, targets)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
    def build_loss(self):
        net = self.net['out']
        prediction = lasagne.layers.get_output(net)
        prediction = T.clip(prediction, 1e-9, 1 - 1e-9)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        loss = loss.mean() + self.lambda2 * regularization.regularize_network_params(net, regularization.l2)
        
        params = lasagne.layers.get_all_params(net, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=self.learning_rate,momentum = 0.9)

         
        test_prediction = lasagne.layers.get_output(net, deterministic=True)
        test_prediction = T.clip(test_prediction, 1e-9, 1 - 1e-9)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                self.target_var)
        
        test_loss = test_loss.mean() + self.lambda2 * regularization.regularize_network_params(net, 
                                                                                               regularization.l2)
        
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        
        train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
        pred_fn = theano.function([self.input_var], lasagne.layers.get_output(net,deterministic=True))
        val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])
        
        return train_fn,val_fn,pred_fn
        
    def fit(self,X,y):
        print("Starting training...")
        X_train, X_test, y_train, y_test = utils.train_test_split(X,y)
        # We iterate over epochs:
        num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            X_train_mini, X_val, y_train_mini, y_val = utils.train_test_split(X_train,y_train,42)
            for batch in self.iterate_minibatches(X_train_mini, y_train_mini, self.batch_size, shuffle=True):
                inputs, targets = batch
                
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            training_loss = train_err / train_batches
            validation_loss = val_err / val_batches
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("Interim results, validated on %s examples" % (len(y_val)))
            print("  training loss:\t\t{:.6f}".format(training_loss))
            print("  validation loss:\t\t{:.6f}".format(validation_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            self.t_loss.append(training_loss)
            self.v_loss.append(validation_loss)
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(X_test, y_test, self.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("\nFinal results, tested on %s examples" % (len(y_test)))
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
    
    def predict_proba(self,X_test):
        proba = []
        N = max(1,self.batch_size)
        for x_chunk in [X_test[i:i + N] for i in range(0, len(X_test), N)]:
            chunk_proba = self.pred_fn(x_chunk)
            for p in chunk_proba:
                proba.append(p)
        return np.array(proba)
    
    def predict(self,X_test):
        proba = self.predict_proba(X_test)
        y_pred = np.argmax(proba,axis=1)
        return np.array(y_pred)
      
    def save_model(self,weights_only = False):
        model_name = self.model_name + '.pkl'
        with open(model_name, 'wb') as f:
            if not weights_only:
                pickle.dump(self, f, -1)
            else:
                w = lasagne.layers.get_all_param_values(self.net['out'])
                pickle.dump(w, f, -1)
            
    def load_model(self, weights_only = False):
        model_name = self.model_name + '.pkl'
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
            w = model['out']['param values']
            f.close()
        if not weights_only:
            self = model
        else:
            lasagne.layers.set_all_param_values(self.net, w)
        