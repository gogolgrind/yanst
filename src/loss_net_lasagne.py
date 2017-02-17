'''
    src:
        https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
        https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg19.py
        https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer
'''


import lasagne

import utils
import pickle
import os
import numpy as np
import seaborn as sns
from models_lasagne import Models as models


from matplotlib import gridspec

class Loss_net():
    
    def __init__(self,model_name = 'vgg19',imshape = (128,128),channel = 3):
        self.net = {}
        self.IMAGE_H = imshape[0]
        self.IMAGE_W = imshape[1]
        self.channel = channel
        self.model_name = model_name
        self.build_model()
        
    def build_model(self):
        if self.model_name == 'vgg19':
            self.net = models.build_vgg19(shape = (self.channel,self.IMAGE_H,self.IMAGE_W))
        elif self.model_name == 'vgg16':
            self.net = models.build_vgg16(shape = (self.channel,self.IMAGE_H,self.IMAGE_W))
            
    def load_weights(self,url = 'https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl'):
        # Download the normalized pretrained weights from:
        # https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl
        # (original source: https://bethgelab.org/deepneuralart/)
        fname = url.split('/')[-1]
        fname = os.path.join('..','models',fname)
        if not os.path.exists(fname):
            print('download %s' %(fname))
            utils.download_file(url)
        else:
            print('%s alredy downloaded' %(fname))
        # get VGG weights
        with open(fname, 'rb') as f:
            values = pickle.load(f, encoding='latin-1')['param values']
        # some bug :C
        if self.model_name == 'vgg16':
            values = values[:26]
        lasagne.layers.set_all_param_values(self.net['pool5'], values)
        
    def get_filter_weights(self,layer_name):
        net = self.net
        if 'conv' in layer_name:
            layer = net[layer_name]
            W = layer.W.get_value()
            b = layer.b.get_value()
            filter_weights = [w + bb for w, bb in zip(W, b)]
            return filter_weights,layer.num_filters
        
    def vizualize(self,layer_name,n=16):
        f,num_filters = self.get_filter_weights(layer_name)
        gs = gridspec.GridSpec(n//8, 8)
        for i in range(n):
            g = gs[i]
            ax = sns.plt.subplot(g)
            ax.grid()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(f[i][0])      
            
    def get_layers(self):
        layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        layers = {k: self.net[k] for k in layers}
        return layers
    