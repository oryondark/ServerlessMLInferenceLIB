from ml_inference.normalization import BatchNorm
import theano.tensor as T
import torch.nn as nn
import torch
import numpy as np

class TorchTest(nn.Module):
    def __init__(self, num_feature):
        super(TorchTest, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_feature)

    def forward(self, x):
        return self.batch_norm(x)


def comparison_withTorch():
    x = np.random.rand(1,3,3,3)
    num_feature = x.shape[-1]

    torch_in = torch.Tensor(x)
    model = TorchTest(num_feature)

    out = model(torch_in)
    torch.save(model.state_dict(), 'state_dict.pth')


    reload_model = TorchTest(num_feature)
    reload_model.load_state_dict(torch.load('state_dict.pth'))
    
