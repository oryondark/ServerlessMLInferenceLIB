import h5py as h5
import theano
import json

class Compile():
    def __init__(self, filePath):
        self.group = h5.File(filePath, 'r')
        #self.graph_flow = {}
        #self.sort_by_graph()

    def sort_by_graph(self):
        nodes = self.group.keys()
        for node in nodes:
        	name = node
        	layer = node.split('_')[1]
        	if len(node.split('_')) > 3:
        		layer = node.split('_')[2]
        	idx = node.split('_')[-1]
        	values = [name, layer, node]
        	self.graph_flow[int(idx)] = values
