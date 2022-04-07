import numpy as np
from Layers import *

class Net():
    def __init__(self):
        self.network = []
        pass
    def add(self,graphnode):
        self.network.append(graphnode)

    def forward(self, X):
        activations = []
        input = X
        #Loop with Initial Input and calculate forward propagation for each layer
        for node in self.network:
            activations.append(node.forward(input))
            input = activations[-1]
        
        assert len(activations) == len(self.network)
        return activations

    def backward(self, layer_inputs, loss_grad):
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = (layer.backward(layer_inputs[layer_index],loss_grad)) #grad w.r.t. input, also weight updates
        

