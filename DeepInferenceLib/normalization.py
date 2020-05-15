import theano.tensor as T
from .layer import Layer
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

import time

class BatchNorm(Layer):
    '''
    Batch Normalization be trained by Gamma(scale factor) and Beta(shift factor).
    expected_y = r * norm(x) + b
    Our library extracts a weights per layer from any deep learning framework,
    weights list involved Batch Normalization layer with scale factor and shift factor.
    '''
    def __init__(self, name="", running_mean=None, running_var=None, scale_factor=None, shift_factor=None, num_features=None, eips=1e-08):
        #In this class to instance, you can optional use argument "name".
        super(BatchNorm, self).__init__(name)
        self.name = name
        self.scale = scale_factor
        self.shift = shift_factor
        self.running_mean = running_mean
        self.running_var = running_var
        self.eips = eips
        self.num_features = num_features

    def _compute(self, x):
        start_time = time.time()
        #if not isinstance(x, np.ndarray):
        #    x = x.eval()
        #scale = self.scale.reshape((1, x.shape[1], 1, 1))
        #shift = self.shift.reshape((1, x.shape[1], 1, 1))
        out = theano.tensor.nnet.bn.batch_normalization_test(x,
                                                        self.scale.reshape((1, int(self.num_features), 1, 1)),
                                                        self.shift.reshape((1, int(self.num_features), 1, 1)),
                                                        self.running_mean.reshape((1, int(self.num_features), 1, 1)),
                                                        self.running_var.reshape((1, int(self.num_features), 1, 1)))
        #out = scale * x + shift
        end_time = time.time()
        #out = theano.tensor.nnet.bn.batch_normalization_test(x, self.scale.reshape((1, x.shape[1], 1, 1)), self.shift.reshape((1, x.shape[1], 1, 1)), self.running_mean.reshape((1, x.shape[1], 1, 1)), self.running_var.reshape((1, x.shape[1], 1, 1)))

        print("Batch Norm Testing Time : {}".format(end_time - start_time))
        return out
