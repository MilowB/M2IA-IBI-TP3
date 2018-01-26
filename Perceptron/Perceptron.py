from Neuron import *

class Perceptron:
    def __init__(self, size, e):
        self.neurons = [Neuron(i, size, e) for i in range(0,10)]

    def activity(self, data, label):
        index = (label == 1).nonzero()
        for n in self.neurons:
            n.new_weight(data, index)

    def predict(self, data):
        max = -1000
        argmax = None
        for n in self.neurons:
            value = n.result(data)
            if max < value or argmax is None:
                max = value
                argmax = n
        return self.neurons.index(argmax)