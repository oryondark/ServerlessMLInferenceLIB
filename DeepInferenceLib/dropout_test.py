from ml_inference.dropout import DropOut
import numpy as np

x = np.random.rand(1,3,3,3)
def compare_sp_dropout_withTorch(x):
    import torch

    dropout = DropOut(x, 'drop_out_test')
    dropout = dropout.forward_and_return()

    torch_in = torch.Tensor(x)
    torch_out = torch.nn.Dropout2d()(torch_in)

    #dropout은
