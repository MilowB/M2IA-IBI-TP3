import gzip
import sys
sys.path.append("./DeepNetwork")
sys.path.append("./Perceptron")
sys.path.append("./ShallowNetwork")


import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import cPickle
import torch
from Perceptron import *
from DeepNetwork import *
from ShallowNetwork import *

def test(learn_data,learn_data_label, test_data, test_data_label,time,network):
    total = 0
    gagne = 0
    for d in range(time):
        d=d%len(learn_data)
        data = torch.from_numpy(learn_data[d])
        label = learn_data_label[d]
        network.activity(data, label)

    for x in range(len(test_data)):
        data = torch.from_numpy(test_data[x])
        label = test_data_label[x]
        p= network.predict(data)
        if int(label[p]) == 1:
            gagne += 1
        total += 1
    return (float(gagne) / float(total)) * 100

def display_shallow():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [1,2,4,8,16,32,64,128]
    time = [5000*i for i in range(1,15)]
    steps = [0.0001, 0.0005, 0.001,0.005, 0.01,0.05,0.1,0.5,1]

    res=[]
    for s in steps:
        network = ShallowNetwork(len(data[0][0][0]), 16, 10, s)
        res.append(test(data[0][0], data[0][1], data[1][0],data[1][1],60000,network))
        print res
    plt.plot(steps, res,'-o')
    plt.ylabel("Taux de reussite")
    plt.xscale("log")
    plt.xlabel("Taux d'apprentissage")
    plt.show()

def display_deep():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [2,4,8,16,32,64,128]#2
    steps = [0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]#0.25
    time = [10000*i for i in range(1,10)] #100 000
    res = []
    #layers = [[20 for i in range(j)] for j in range(1,10)]
    #number = [ i for i in range(1,10)]
    hid = [ [i,i] for i in hiddens]
    for h in hid:
        # Dimension des couches cachees
        network = DeepNetwork( len(data[0][0][0]), 10, h,0.25,100000)
        network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
        res.append(network.test( data[1][0], data[1][1]))
    plt.plot(hiddens, res,"-o")
    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de neurones par couches")
    plt.show()


def display():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    steps = [0.0001, 0.0005, 0.001,0.005, 0.01,0.05,0.1]
    time = [5000*i for i in range(1,15)] # defaut = 63 000
    res = []
    for i in time:
        network = Perceptron(len(data[0][0][0]), 0.005)
        res.append(test(data[0][0], data[0][1], data[1][0],data[1][1],i,network))
    plt.plot(time, res,'-o')
    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Pas d'apprentissages")
    plt.show()

if __name__ == '__main__':
    display_deep()