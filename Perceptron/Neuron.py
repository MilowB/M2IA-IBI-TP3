import torch
import math

class Neuron:
    def __init__(self, id, size):
        self.ti = torch.zeros(10)
        self.ti[id] = 1
        self.e = 0.005
        self.weight = torch.rand(size)
        test = torch.rand(size)

    def new_weight(self, data, index):
        data = torch.from_numpy(data)
        act = torch.dot(self.weight, data)
        delta = self.e * data * (self.ti[index] - act)
        self.weight = self.weight + delta

    def result(self, data):
        data = torch.from_numpy(data)
        return torch.dot(self.weight, data)