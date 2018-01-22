import sys
sys.path.append("../")

import torch
import cPickle
from ShallowNetwork import *
from lecture_data import *

def main():
    data = cPickle.load(gzip.open('../mnist.pkl.gz'))
    apprentissage = data[0]
    test = data[1]
    
    size = len(data[0][0][0])
    hiddenlayer = 2
    exitlayer = len(apprentissage[1][0])
    #e= 0.0001
    e = 0.1
    network = ShallowNetwork(size, hiddenlayer, exitlayer, e)

    total = 0
    gagne = 0
    print("Apprentissage")
    for x in range(len(apprentissage[0])):
        d =  x % len(apprentissage[0])
        data = torch.from_numpy(apprentissage[0][d])
        label = apprentissage[1][d]
        network.activity(data, label)

    print("Tests")
    for x in range(len(test[0])):
        data = torch.from_numpy(test[0][x])
        label = test[1][x]
        if int(label[network.predict(data,label)]) == 1:
            gagne += 1
        total += 1
    print (float(gagne) / float(total)) * 100, "%"


if __name__ == '__main__':
    main()
