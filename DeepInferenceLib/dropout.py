import theano.tensor as T
from .layer import Layer
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

class DropOut(Layer):
    '''
    Theano did not implementation of layer for dropout.
    Serverless Computing need not high-computing to processing multiple dataset.
    Therefore, I think the dropout just enough to make simply on python platform.
    '''
    def __init__(self,name="", p=0.5, rescale=True, shared_axex=(), algorithm='spatial',**kwargs):
        #In this class to instance, you can optional use argument "name".
        super(DropOut, self).__init__(name)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self._p = p
        self._rescale = rescale
        self.shared_axex = tuple(shared_axex)
        self.algorithm = algorithm
        self.mask = None

    def forward_and_return(self):
        if self.algorithm == 'spatial':
            return self._spatial_dropout()

    def _spatial_dropout(self):
        """
        [1] J. Tompson, R. Goroshin, A. Jain, Y. LeCun, C. Bregler (2014):
            Efficient Object Localization Using Convolutional Networks.
            https://arxiv.org/abs/1411.4280
        """

        #incomming argument is a layer involved 'output_shape', getattr function will to return shape.
        ndim = len(getattr(self.incomming, 'output_shape', self.incomming.shape))
        # x_shape = (1,2,3,4) => shared_axex = (2, 3)
        self.shared_axex = tuple(range(2, ndim))
        if self._p == 0:
            return self.incomming
        if not isinstance(self.incomming, tuple):
            return self._compute(self.incomming)

    def _compute(self, input):
        '''
        retain_prob = 1 - self._p
        if self._rescale == True:
            input /= retain_prob

        mask_axes = input.shape
        if self.shared_axex:
            shared_axex = tuple( a if a >= 0 else a + len(input.shape) for a in self.shared_axex)
            mask_axes = tuple( 1 if a in self.shared_axex else x for a, x in enumerate(mask_axes))

        self.mask = self._srng.binomial(mask_axes, p=retain_prob, dtype=theano.config.floatX).eval()
        return (input * self.mask) * 2 # Torch에서 값이 두배가 되는데, 이유는 모르겠음....
        '''
        return input

    def get_out(self, input):
        return input

    def get_param(self):
        print_dict = {}
        print_dict['shape'] = self.input_shape
        print_dict['dropout_rate'] = self._p
        print_dict['algorithm'] = self.algorithm
        print_dict['shared_axex'] = self.shared_axex
        print_dict['name'] = self.name
        print_dict['in_value'] = self.incomming
        print(print_dict)
