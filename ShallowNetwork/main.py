import sys
sys.path.append("../")

import torch,torch.utils.data
import cPickle
from ShallowNetwork import *
from lecture_data import *

def main():
    TRAIN_BATCH_SIZE = 1
    data = cPickle.load(gzip.open('../mnist.pkl.gz'))
    apprentissage = data[0]
    test = data[1]
    size = len(data[0][0][0])
    hiddenlayer = 5
    exitlayer = len(apprentissage[1][0])
    #e= 0.0001
    e = 0.01
    network = ShallowNetwork(size, hiddenlayer, exitlayer, e)

    total = 0
    gagne = 0
    print("Apprentissage")
    for x in range(20000):
        #(_,(image,label)) = enumerate(train_loader).next()

        d =  x % len(apprentissage[0])
        data = torch.from_numpy(apprentissage[0][d])
        label = apprentissage[1][d]
        lr = network.activity(data, label)


    print("Tests")
    dict = {}
    for x in range(6000):
        x = x % len(test[0])
        data = torch.from_numpy(test[0][x])
        label = test[1][x]
        pred = network.predict(data)
        if int(label[pred])== 1:
            gagne += 1
        total += 1
    print (float(gagne) / float(total)) * 100, "%"


if __name__ == '__main__':
    main()
