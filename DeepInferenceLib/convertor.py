import h5py
import numpy as np
from .torch_linker import *
import json

def weight_parser(deepModel, framework, torch_dummy):

    if framework == "torch":
        #create_metadata(deepModel, framework, torch_input_shape)
        torch_parser(deepModel, framework, torch_dummy)


    #elif framework == 'tensorflow'
    #elif framework == 'keras'
    #elif framework == 'caffe2'

#추출한 가중치들을 Numpy 배열형태로 변환 후, h5포맷으로 저장
layer = {
    "BatchNorm2d": [["running_mean", "running_var", "weight", "bias"], ["num_features"]],
    "Conv2d" : [["weight", "bias"], ["padding", "stride"]],
    "MaxPool2d": [[], ["kernel_size", "padding", "stride"]],
    "Dropout2d": [[], []],
    "Linear": [["weight", "bias"], []],
    "Softmax": [[],[]]
}
# 추가해야함.
Activation = {
    "ReluBackward0" : "ReLu",
}
def torch_parser(model, tool, dummy):

    graph_flow = {}
    with h5py.File('weights_dict.h5', 'w') as f:

        def printnorm(self, input, output):
            name = None
            act = None
            try:
                print('backward_fn: {}\n'.format(input[0].grad_fn.name()))
                name = input[0].grad_fn.name()
                act = Activation[name]
            except:
                print('backward_fn: {}\n'.format('none'))

            if len(graph_flow.keys()) > 0:
                max_key = max(graph_flow.keys())
                key = max_key + 1
            else:
                key = 0

            meta_set = {"name":self.__class__.__name__, "pre_activation":act}

            g = f.create_group(str(key))
            layer_meta = layer[self.__class__.__name__]
            for v in layer_meta[0]:
                parameter = getattr(self, v, None)
                print(parameter)
                print(v)
                if parameter != None:
                    #print("get weights")
                    h5_parm = g.create_dataset(v, np.array(parameter.detach().numpy()).shape)
                    h5_parm[:] = np.array(parameter.cpu().detach().numpy(), dtype=np.float64)

            for v in layer_meta[1]:
                parameter = getattr(self, v, None)
                if parameter != None:
                    meta_set[v] = parameter

            graph_flow[int(key)] = meta_set
        odict_item = model._modules.items()
        for _, layer_cls in list(odict_item):
            layer_cls.register_forward_hook(printnorm)
        model(dummy)

    #Graph_flow save
    with open('graph_flow.meta', 'w') as f:
        graph_flow['tool'] = tool
        json.dump(graph_flow, f)
