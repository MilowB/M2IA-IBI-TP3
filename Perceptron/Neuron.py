import torch
import math

class Neuron:
    def __init__(self, id, size, e):
        self.ti = torch.zeros(10)
        self.ti[id] = 1
        self.e = e
        # +1 pour le biais
        self.weight = torch.rand(size + 1)

    def new_weight(self, data, index):
        data = torch.cat((torch.Tensor([1]), data), 0)
        act = torch.dot(self.weight, data)
        delta = self.e * data * (self.ti[index] - act)
        self.weight = self.weight + delta

    def result(self, data):
        data = torch.cat((torch.Tensor([1]), data), 0)
        return torch.dot(self.weight, data)