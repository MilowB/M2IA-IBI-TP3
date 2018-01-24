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
    #e= 0.0001
    e = 0.1

    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])

    network = deepNetwork(sizeIn, sizeOut, hiddenlayer, torch.from_numpy(apprentissage[0]), test_data, test_data_label)

    '''
    total = 0
    gagne = 0
    for x in range(10000):
        d =  x % len(apprentissage[0])
        data = torch.from_numpy(apprentissage[0][d])
        label = apprentissage[1][d]
        network.activity(data, label)

    for x in range(len(test[0])):
        data = torch.from_numpy(test[0][x])
        label = test[1][x]
        if int(label[network.predict(data,label)]) == 1:
            gagne += 1
        total += 1
    print (float(gagne) / float(total)) * 100, "%"
    '''

if __name__ == '__main__':
    main()
