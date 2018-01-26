import sys
sys.path.append("../")

import torch
import cPickle
from DeepNetwork import *
from lecture_data import *

def main():
    data = cPickle.load(gzip.open('../mnist.pkl.gz'))
    apprentissage = data[0]
    test = data[1]
    
    sizeIn = len(data[0][0][0])
    hiddenlayer = 100
    sizeOut = len(apprentissage[1][0])
    e = 5e-6

    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])

    #Dimension des couches cachees [10, 10]
    network = DeepNetwork([100, 10, 100], sizeIn, sizeOut, e, True)
    network.train(torch.from_numpy(apprentissage[0]), torch.from_numpy(apprentissage[1]))
    network.test(test_data, test_data_label)


if __name__ == '__main__':
    main()
