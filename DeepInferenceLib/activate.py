import theano.tensor as T

def ReLu(x):
    y = T.maximum(0.0, x)
    return(y)

def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
