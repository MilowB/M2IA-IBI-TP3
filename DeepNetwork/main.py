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
    sizeOut = len(apprentissage[1][0])
    #e = 5e-2
    e = 0.01

    # images de la base de test
    test_data = data[1][0]
    # labels de la base de test
    test_data_label = data[1][1]

    #Dimension des couches cachees [10, 10]
    network = DeepNetwork(sizeIn, sizeOut, [32], e, 20000, batch_number=1)
    network.train_optimize(torch.from_numpy(apprentissage[0]), torch.from_numpy(apprentissage[1]))
    acc = network.test_optimize(test_data, test_data_label)
    print acc


if __name__ == '__main__':
    main()
