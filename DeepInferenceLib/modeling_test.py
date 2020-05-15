from ml_inference.dropout import * #DropOut
from ml_inference.normalization import * #BatchNorm
from ml_inference.compiler import * # copile neuralnet model from h5py.


import theano
import theano.tensor as T

from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv



'''
Notice Using theano framework have to change "channel first" of shape if you serve an array 3D shape more.
'''
# Usage modeling. In this case Net will be returned your neural network graph ordering to you.
class Net(Compile):
    import h5py
    import numpy as np
    def __init__(self, file_path):
        super(Net, self).__init__(file_path)

    def get_graph_flow(self):
        print(self.graph_flow)
        return self.graph_flow

    def get_groups(self):
        return self.group


class modeling_compute_testing():
    import numpy as np
def register_hook(module):
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)
        m_key = "%s-%i" % (class_name, module_idx + 1)
        summary[m_key] = OrderedDict()
        summary[m_key]["input_shape"] = list(input[0].size())
        summary[m_key]["input_shape"][0] = batch_size
        if isinstance(output, (list, tuple)):
            summary[m_key]["output_shape"] = [
                [-1] + list(o.size())[1:] for o in output
            ]
        else:
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = batch_size
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        summary[m_key]["nb_params"] = params
    if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
    ):
        hooks.append(module.register_forward_hook(hook))
