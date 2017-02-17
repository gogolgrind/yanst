'''
    src:
        https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer
'''

import lasagne
import theano
import theano.tensor as T
import losses_lasagne as losses
import numpy as np
import scipy as sp
from tqdm import tqdm

class Fast_style_transfer():
    """
    Todo Texture nets
    """
    
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def transform(self):
        pass
    
class Slow_style_transfer():
    """
    Main class for
    Neural Style transfer based on  https://arxiv.org/abs/1508.06576
    
    special params:
        init_type {1,2,3} - type of inialization random, content based or style based
        gray_scale_only boolean  - costraint to rgb colors
        
    """
    def __init__(self,layers,IMAGE_H,IMAGE_W,content,style,gray_only, \
                 channel = 3,init_type = 2, cont_loss_type = 'frob', \
                 content_weight = 0.001, style_weight =  0.2e7, tv_weight = 1e-12):
        self.layers = layers
        self.channel = channel
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.CONTENT_WEIGHT = content_weight
        self.STYLE_WEIGHT =  style_weight
        self.TV_WEIGHT = tv_weight
        self.hist = []
        self.gray_only = gray_only
        self.content = content
        self.cont_loss_type = cont_loss_type
        self.style = style
        if init_type == 0:
            im = np.random.uniform(0, 255, (1, 3, self.IMAGE_H,self.IMAGE_W)) - 128.
            im = im/255
            self.generated_image = theano.shared(im.astype('float32'))
        elif init_type == 1:
            self.generated_image = theano.shared(np.float32(content.copy()))
        elif init_type == 2:
            self.generated_image = theano.shared(np.float32(style.copy()))
        self.total_loss = None
        
    def get_losses(self):
        """
            generate loss function for optimisation
        """
        layers = self.layers
        content = self.content
        style = self.style
       
        input_im_theano = T.tensor4()
        outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
        outputs = np.array(outputs)
        
        style_features = {k: 
                          theano.shared(output.eval({input_im_theano: style}))
                        
                          for k, output in zip(layers.keys(), outputs)}

   
        
        gen_features = lasagne.layers.get_output(layers.values(), self.generated_image)
        gen_features = {k: v for k, v in zip(self.layers.keys(), gen_features)}
       
        temp_losses = []
           
        if self.cont_loss_type == 'frob':
            im_diff =  content - self.generated_image.get_value().astype('float32')
            temp_losses.append(self.CONTENT_WEIGHT * ( theano.shared( sp.linalg.norm (im_diff))))
        else:
            content_features = {k: theano.shared(output.eval({input_im_theano: content}))
                          for k, output in zip(layers.keys(), outputs)}
            temp_losses.append(self.CONTENT_WEIGHT * losses.content_loss(content_features, gen_features, 'conv4_2'))
        
        # style loss
        for lr in ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']:
            temp_losses.append(self.STYLE_WEIGHT * losses.style_loss(style_features, gen_features, lr))
        # total variation penalty
        temp_losses.append(self.TV_WEIGHT * losses.total_variation_loss(self.generated_image))
        self.total_loss = np.sum(temp_losses)

    def fit_transform(self,iterations):
        """
            methods for periodic image stylisation, return array of images 
        """
        self.get_losses()
        x0 = self.generated_image.get_value().astype('float32')
        xs = []
        xs.append(x0)
        eta = T.scalar(name='learning_rate')
      
        updates = lasagne.updates.adam(self.total_loss,[self.generated_image],eta)
        self.opt = theano.function([eta], self.total_loss, updates=updates)
        base_eta = 20
        eta_decay = 0.7
        for it in range(iterations):
            if it > 0 and it  % 100 == 0:
                base_eta = base_eta * eta_decay 
            print(('\niterations : %s/%s') % (it+1,iterations))
            epochs = 40
            mean_loss = 0
            for ep in tqdm(range(epochs),ascii = True):
                current_loss = self.opt(base_eta)
                mean_loss += current_loss
                x0 = self.generated_image.get_value().astype('float32')
                self.generated_image.set_value(x0)
            mean_loss = mean_loss/epochs
            self.hist.append(mean_loss)
            print('\n','mean loss per iteration',mean_loss)
            if self.gray_only:
                im = x0[0]
                k = np.argmax([sp.linalg.norm(im[0]),
                               sp.linalg.norm(im[1]),
                               sp.linalg.norm(im[2])])
                x0 = np.array((im[k],im[k],im[k]))[np.newaxis,...]
            xs.append(x0)
        return xs