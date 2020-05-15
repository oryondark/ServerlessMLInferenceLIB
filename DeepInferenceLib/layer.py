import theano
import theano.tensor as T
import numpy as np

from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d, conv
from theano.tensor.nnet.nnet import softmax

class Layer(object):
    def __init__(self, name=None):
        # 입력 값이 튜플일 경우, 절대로 입력 피처가 아님.
        # 근본적인 이유로, Tuple 데이터는 수정이 불가능하기 때문에, 이를 list로 변경하는 등 복잡함.
        # 따라서 딥러닝은 입력 피처나 가중치를 항상 수정 가능한 타입으로 제공
        #if isinstance(incomming, tuple):
        #    raise "It can not initialze this module because that is not to implement the model"
        if name == None:
            raise "Must be named this layer"
        #self.incomming = incomming
        #self.input_shape = incomming.shape
        self.name = name

    def get_param(self):
        '''
        In this function will to return a parameter values this layer.
        So, you can check the parameter to debug type or value the layer.
        Note that you will to make the any hidden-layer, should be implemented 'Layer'.
        But maybe this don't return parameter values when use to implement DropOut.
        '''
        return var_list


'''
kernel -> stride -> padding
'''
class Conv2D():
    def __init__(self, name=None, kernel=None, bias=None, stride=None, padding=None, kernel_size=None):
        try:
            print('bias : ', bias.shape)
            print('\nkernel : ', kernel.shape)
            print('\nstride : ', stride.shape)
        except:
            pass

        self.kernel = theano.shared(np.asarray(kernel, dtype=theano.config.floatX), borrow=True, name=name+"_w")
        self.kernel_size = kernel.shape
        if stride:
            self.stride = tuple(stride)
        if padding:
            self.padding = tuple(padding)
        if isinstance(bias, np.ndarray):
            self.bias = theano.shared(np.asarray(bias, dtype=theano.config.floatX), borrow=True, name=name+"_b")
        else:
            self.bias = []

        self.init = False

    def _compute(self, x):
        x = conv2d(x, filters=self.kernel, filter_shape=self.kernel.get_value().shape, border_mode=self.padding, subsample=self.stride)
        if self.bias:
            x = x + self.bias.dimshuffle('x', 0, 'x', 'x')
        return x#.eval()


class Pool2D():
    def __init__(self, name=None, stride=None, kernel=None, padding=None):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.name = name

    def _compute(self, x):
        x = pool_2d(x, ds=(self.kernel, self.stride), ignore_border=False)
        return x#.eval()

class Linear():
    def __init__(self, name=None, weight=None, bias=None):
        self.weight = weight
        self.name = name
        self.bias = bias
        if self.bias == None:
            self.bias = np.zeros(self.weight.shape[0])

    def _compute(self, x):
        x = np.dot(x, self.weight.T) + self.bias
        return x


class SoftMax():
    def __init__(self, name=None):
        self.name = name

    def _compute(self, x):
        x = softmax(x)
        return x#.eval()
