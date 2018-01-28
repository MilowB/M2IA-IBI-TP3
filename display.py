import gzip
import sys
import numpy as np
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

def display_relu():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [2,4,8,16,32,64,90,128]#2
    arrRelu = []

    for i in range(8):
        relu = []
        for h in hiddens:
            network = DeepNetwork( len(data[0][0][0]), 10, [h], 1e-2, 10000, "relu")
            network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
            relu.append(network.test( data[1][0], data[1][1]))
        if i > 0:
            arrRelu.append(relu)

    for arr in arrRelu:
        relu = np.add(relu, arr)
    if len(arrRelu) >= 1:
        relu = relu / (len(arrRelu) + 1)

    fig, ax = plt.subplots()
    ax.plot(hiddens, relu, "-o", label='ReLU')
    ax.legend(loc='center right', shadow=True)

    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de neurones par couches")
    plt.savefig("relu.png")
    plt.show()

def display_tanh_pas():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [90]#2
    steps = [1e-6, 1e-5, 1e-4,1e-3, 1e-2,1e-1]
    
    arrTanh = []

    for i in range(1):
        tanh = []
        for h in steps:
            network = DeepNetwork( len(data[0][0][0]), 10, [90], h, 1000, "tanh")
            network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
            tanh.append(network.test( data[1][0], data[1][1]))
        if i > 0:
            arrTanh.append(tanh)

    for arr in arrTanh:
        tanh = np.add(tanh, arr)
    if len(arrTanh) >= 1:
        tanh = tanh / (len(arrTanh) + 1)

    fig, ax = plt.subplots()
    ax.plot(steps, tanh, "-o", label='Tanh')
    ax.legend(loc='center right', shadow=True)

    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de neurones par couches")
    plt.savefig("tanh.png")
    plt.show()

def display_relu_pas():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [90]#2
    steps = [1e-6, 1e-5, 1e-4,1e-3, 1e-2,1e-1]
    
    arrRelu = []

    for i in range(1):
        relu = []
        for h in steps:
            network = DeepNetwork( len(data[0][0][0]), 10, [90], h, 10000, "relu")
            network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
            relu.append(network.test( data[1][0], data[1][1]))
        if i > 0:
            arrRelu.append(relu)

    for arr in arrRelu:
        relu = np.add(relu, arr)
    if len(arrRelu) >= 1:
        relu = relu / (len(arrRelu) + 1)

    fig, ax = plt.subplots()
    ax.plot(steps, relu, "-o", label='ReLU')
    ax.legend(loc='center right', shadow=True)

    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de neurones par couches")
    plt.savefig("relu.png")
    plt.show()

def display_activations():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    hiddens = [2,4,8,16,32,64,128]#2
    relu = []
    tanh = []
    sigmoid = []

    for h in hiddens:
        network = DeepNetwork( len(data[0][0][0]), 10, [h], 1e-2, 20000, "relu")
        network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
        relu.append(network.test( data[1][0], data[1][1]))
    
    for h in hiddens:
        network = DeepNetwork( len(data[0][0][0]), 10, [h], 1e-2, 50000, "tanh")
        network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
        tanh.append(network.test( data[1][0], data[1][1]))
    
    for h in hiddens:
        network = DeepNetwork( len(data[0][0][0]), 10, [h], 0.1, 60000, "sigmoid")
        network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
        sigmoid.append(network.test( data[1][0], data[1][1]))

    fig, ax = plt.subplots()
    ax.plot(hiddens, tanh, "-o", label='Tanh')
    ax.plot(hiddens, relu, "-o", label='ReLU')
    ax.plot(hiddens, sigmoid, "-o", label='Sigmoid')
    ax.legend(loc='center right', shadow=True)

    plt.xscale("linear")
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de neurones par couches")
    plt.savefig("deep.png")
    plt.show()

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

def display_optim():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    layers = [[20]*j for j in range(1,8)]
    res=[]
    for l in layers:
        # Dimension des couches cachees
        network = DeepNetwork(len(data[0][0][0]), 10, l,0.1 ,40000)
        network.train(torch.from_numpy(data[0][0]), torch.from_numpy(data[0][1]))
        res.append(network.test(data[1][0], data[1][1]))
        print res
#[62.58571428571429, 68.41428571428571, 74.88571428571429, 77.34285714285714, 83.75714285714285, 44.7, 20.085714285714285, 9.685714285714287, 9.685714285714287]
#[10.042857142857143, 11.785714285714285, 41.37142857142857, 72.47142857142858, 68.01428571428572, 82.07142857142857, 81.47142857142858, 50.91428571428571, 30.47142857142857]
#[10.942857142857143, 10.842857142857143, 10.842857142857143, 10.842857142857143, 10.842857142857143, 10.085714285714285, 10.085714285714285, 10.085714285714285, 10.942857142857143]

#[18.242857142857144, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142]
#[92.10000000000001, 89.72857142857143, 68.34285714285714, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142]


def graph_optim():
    steps = [j for j in range(1,8)]
    adam = [88.02857142857144, 88.38571428571429, 80.0, 18.728571428571428, 10.042857142857143, 10.042857142857143, 10.042857142857143]
    adagrad = [92.10000000000001, 89.72857142857143, 68.34285714285714, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142]
    norm = [89.77142857142857,18.242857142857144, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142, 10.242857142857142]

    plt.plot(steps, adam,"-o",label="Adam")
    plt.plot(steps, adagrad, "-o", label="Adagrad")
    plt.plot(steps, norm, "-o", label="Aucun")
    plt.xscale("linear")
    plt.legend(loc='upper right')
    plt.ylabel("Taux de reussite")
    plt.xlabel("Nombre de couches cachees")
    plt.show()

if __name__ == '__main__':
    display_activations()
