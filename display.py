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
    hiddens_number = [1,2,4,8,16,32,64]
    hiddens = [1,2,4,8,16,32,64,128]
    steps = [0.0001, 0.0005, 0.001,0.005, 0.01,0.05,0.1,0.5,1]
    time = [10000*i for i in range(1,8)]
    for j in time:
        for hidden in hiddens:
            res =[]
            # Dimension des couches cachees
            network = DeepNetwork([10,], sizeIn, sizeOut, 5e-4, False)
            network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
            network.test( data[1][0], data[1][1])
            res.append(test(data[0][0], data[0][1], data[1][0],data[1][1],j,network))
    plt.plot(time, res)
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de couches cachees")
    plt.show()

    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue'])
    bounds = [i for i in range(0,10)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, 10, 1));
    ax.set_yticks(np.arange(-.5, 10, 1));

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
    display_shallow()