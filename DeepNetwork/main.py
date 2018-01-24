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

    hiddenlayer = 10
    sizeIn = len(apprentissage[0][0])
    sizeOut = len(apprentissage[1][0])
    #e= 0.0001
    e = 0.1

    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])

    network = deepNetwork(sizeIn,
                          sizeOut,
                          hiddenlayer,
                          torch.from_numpy(apprentissage[0]),
                          torch.from_numpy(apprentissage[1]),
                          test_data,
                          test_data_label)


if __name__ == '__main__':
    main()
