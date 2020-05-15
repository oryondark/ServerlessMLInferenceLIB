from ml_inference.dropout import * #DropOut
from ml_inference.normalization import * #BatchNorm
from ml_inference.compiler import * # copile neuralnet model from h5py.
from ml_inference.activate import *
from ml_inference.layer import *


import theano
import theano.tensor as T

#from theano.tensor.signal.pool import pool_2d
#from theano.tensor.nnet import conv
#from theano.tensor.nnet.nnet import softmax

import h5py
import numpy as np
import json
import time

class NeuralNet():
    def __init__(self, group, meta_path):
        #self.story_graph = len(self._get_graph_flow())
        #self.group = self._get_group() # from compiler.py
        self.group = group
        with open(meta_path, 'r') as js:
            self.js = json.load(js)

        self.batchNormList = {}
        self.DropOut = None
        self.toolname = None
        generated_model = self._restore_model(meta_path)
        self.model = generated_model
        self.activation_fns = {
            "ReLu" : ReLu,
            "Sigmoid" : Sigmoid,
        }
        del(self.group)
        del(generated_model)


    def _layer_from_torch(self, name):
        """
        layer 정보 파싱
        """
        layer = {
            "BatchNorm2d" : [BatchNorm, ["running_mean", "running_var", "weight", "bias"], ["num_features"]],
            "Conv2d" : [Conv2D, ["weight", "bias"], ['stride', 'padding', 'kernel_size']],
            "Dropout2d" : [DropOut,[], []],
            "MaxPool2d" : [Pool2D,[], ["stride", "kernel_size", 'padding']],
            "Linear" : [Linear, ["weight", "bias"], []],
            "Softmax" : [SoftMax, [], []],
            "ReLu" : [ReLu, [], []]
        }

        return layer[name]

    def _compute(self, x):
        total_time_incompute = 0
        previous_layer = None
        for seq in self.model:
            per_layer_time = time.time()
            pre_activation = self.js[str(seq)]['pre_activation']
            if pre_activation:
                act_fn = self.activation_fns[pre_activation]
                x = act_fn(x)
            if self.model[seq].__class__.__name__ == "Linear" and previous_layer != "Linear":#len(x.shape) > 2:
                eval_time = time.time()
                print("[TimeMeasurement] Evaluation Time : {}".format(time.time() - eval_time))
                x = theano.tensor.reshape(x, (1,128))# 이 부분도 파싱이 필요한데, 지금은 실험을 위해 하드 코딩.
                print("[TimeMeasurement] compute time : {}".format(time.time() - total_time_incompute))
                return x
            x = self.model[seq]._compute(x)
            previous_layer = self.model[seq].__class__.__name__
            total_time_incompute += time.time()
            print("[TimeMeasurement] {} layer Time : {}".format(self.js[str(seq)]['name'], time.time() - per_layer_time))
        #return x

    def _restore_model(self, meta, activation="ReLu"):

        js_keys = self.js.keys()
        self.tool_name = self.js['tool']
        self.js.pop('tool') # remove key

        model = {}
        for seq_n, key in enumerate(js_keys):
            torch_layer = self.js[key]['name']

            attr = self._layer_from_torch(torch_layer)
            layer_obj = attr[0] # layer object
            args = []
            args.append(str(key))
            if attr[1]:
                for sub_key in attr[1]:
                    if sub_key in self.group[key]:
                        args.append(self.group[key][sub_key].value)
                    else:
                        args.append(None)
            if attr[2]:
                for sub_key in attr[2]:
                    if sub_key in self.js[key]:
                        args.append(self.js[key][sub_key])
                    else:
                        args.append(None)
            print(args)
            model[seq_n] = layer_obj(*args)
            #name, layer, simbol = self.graph.flow[int(i)+1]
        return model

    def _get_graph_flow(self):
        '''
        {sequence_number : [layername, class, groupName]
        '''
        return self.graph_flow

    def _get_group(self):
        return self.group
