
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer

class Models():
    
    @staticmethod
    def build_vgg19(shape):
        net = {}
        net['input'] = InputLayer((None, shape[0], shape[1], shape[2]))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
        return net
    
    @staticmethod
    def build_vgg16(shape):
        net = {}
        net['input'] = InputLayer((None, shape[0], shape[1], shape[2]))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2,mode='average_exc_pad')
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2,mode='average_exc_pad')
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2,mode='average_exc_pad')
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2,mode='average_exc_pad')
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2,mode='average_exc_pad')
        return net
    
    @staticmethod
    def build_vgg16_full(shape,input_var):
        pass
    
    @staticmethod
    def build_roadar_full(shape,input_var):
        net = {}
        #Input layer:
        net['input'] = InputLayer((None, shape[0], shape[1], shape[2]),input_var=input_var)
        #Convolution + Pooling
        net['conv_11'] = ConvLayer(net['input'], num_filters=64,filter_size=5, stride=1)
        net['pool_11'] = PoolLayer(net['conv_11'], 2, stride = 2)
        
        net['conv_12'] = ConvLayer(net['pool_11'], num_filters=64,filter_size=5, stride=1)
        net['pool_12'] = PoolLayer(net['conv_12'], 2, stride = 2)
        
        net['conv_21'] = ConvLayer(net['pool_12'], num_filters=128, filter_size=3)
        net['pool_21'] = PoolLayer(net['conv_21'], pool_size=2)
        #Fully-connected + dropout
        net['fc_1'] = DenseLayer(net['pool_21'], num_units=500)
        net['drop_1'] = DropoutLayer(net['fc_1'],  p=0.5)
        #Output layer:
        net['out'] = DenseLayer(net['drop_1'], num_units=2, 
                                               nonlinearity=lasagne.nonlinearities.softmax)
        return net
    