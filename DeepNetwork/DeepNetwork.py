import torch
from torch import nn
from torch.autograd import Variable
import numpy
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

#AVANT DE COMMENCER VOICI LE LIEN VERS LA DOC:
# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

class DeepNetwork:
    def __init__(self, sizeEntries, sizeOut, dimensions, e, time, function = "relu", batch_number = 1, optimize = True, debug = False):
        self.debug = debug
        self.batch_number = batch_number
        self.optimize = optimize
        self.time = time

        if self.optimize:
            if function == "relu":
                self.unlinear = nn.ReLU
            elif function == "tanh":
                self.unlinear = nn.Tanh
            elif function == "sigmoid":
                self.unlinear = nn.Sigmoid
        else:
            if function == "relu":
                self.unlinear = nn.ReLU()
            elif function == "tanh":
                self.unlinear = torch.tanh
            elif function == "sigmoid":
                self.unlinear = torch.sigmoid

        # dimensions is hidden dimensions
        # D_in is input dimension;
        # D_out is output dimension.
        D_in, D_out = sizeEntries, sizeOut

        if self.optimize:
            #Functions used in the optimizer
            self.model = torch.nn.Sequential(
                nn.Sigmoid(),
                torch.nn.Linear(batch_number, D_out),
            )
            #loss function
            self.loss_fn = torch.nn.MSELoss(size_average=False)

            #optimizer, adagrad adapte la fonction d'activation
            self.optimizer = optim.Adagrad([
                {'params': self.model.parameters()},
            ])

        # Neurones et leurs poids
        resize = 0.13
        wentry = torch.randn(D_in + 1, dimensions[0]).uniform_(-resize, resize).type(dtype)
        self.wentry = Variable(wentry, requires_grad=True) # Couche entree

        if self.optimize:
            optim_hidden = [("e" , torch.nn.Linear(D_in, dimensions[0]))]
        self.whidden = []
        lastSize = dimensions[0]
        for i in range(0, len(dimensions)):
            hiddenLayer = torch.randn(lastSize + 1, dimensions[i]).uniform_(-resize, resize).type(dtype)
            layer = Variable(hiddenLayer, requires_grad=True) # Couche cachee
            self.whidden.append(layer)
            lastSize = dimensions[i]
            if self.optimize:
                optim_hidden.append(("e"+str(i), self.unlinear()))
        if self.optimize:
            optim_hidden.append(("f" , torch.nn.Linear(dimensions[0], D_out)))
        self.wout = Variable(torch.randn(lastSize + 1, D_out).uniform_(-resize, resize).type(dtype), requires_grad=True) # Couche sortie
        self.learning_rate = e

        #Functions used in the optimizer
        if self.optimize:
            self.model = torch.nn.Sequential(OrderedDict(optim_hidden))
            #loss function
            self.loss_fn = torch.nn.MSELoss(size_average=False)
            #optimizer, adagrad adapte la fonction d'activation
            self.optimizer = optim.Adam([
                {
                    'params': self.model.parameters(),
                    'lr': e
                },
            ])
        if self.debug:
            print "learning rate : ", self.learning_rate
            print "number hidden layers : ", len(self.whidden)


    def train(self, data, label):
        if self.optimize:
            return self._train_optimize(data, label)
        else:
            return self._train_classic(data, label)


    def test(self, test_data, test_data_label):
        if self.optimize:
            return self._test_optimize(test_data, test_data_label)    
        else:
            return self._test_classic(test_data, test_data_label)


    def _train_optimize(self, data, label):
        x = Variable(data.type(dtype), requires_grad=False)
        y = Variable(label.type(dtype), requires_grad=False)
        # Forward pass: compute predicted y by passing x to the model.
        batch = self.batch_number
        size = len(data)
        for t in range(self.time):
            st = (t * batch) % size
            end = ((t + 1) * batch) % size
            end = size if end < st else end
            x = Variable(data[st:end].type(dtype), requires_grad=False)
            y = Variable(label[st:end].type(dtype), requires_grad=False)
            y_pred = self.model(x)
            # Compute and print loss.
            loss = self.loss_fn(y_pred, y)
            if self.debug:
                print(t, loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def _train_classic(self, data, label):
        x = Variable(data.type(dtype), requires_grad=False)
        y = Variable(label.type(dtype), requires_grad=False)

        batch = self.batch_number
        size = len(data)

        for t in range(self.time):
            st = (t * batch) % size
            end = ((t + 1) * batch) % size
            end = size if end < st else end
            x = Variable(data[st:end].type(dtype), requires_grad=False)
            x = self._addBiais(x)
            y = Variable(label[st:end].type(dtype), requires_grad=False)
            y_pred = self.unlinear(x.mm(self.wentry))
            y_pred = self._addBiais(y_pred)

            for layer in self.whidden:
                y_pred = self.unlinear(y_pred.mm(layer))
                y_pred = self._addBiais(y_pred)

            y_pred = self.unlinear(y_pred.mm(self.wout))
            loss = (y_pred - y).pow(2).sum()

            if self.debug:
                print(t, loss.data[0])

            loss.backward()
            
            self.wentry.data -= self.learning_rate * self.wentry.grad.data
            for layer in self.whidden:
                layer.data -= self.learning_rate * layer.grad.data
            self.wout.data -= self.learning_rate * self.wout.grad.data

            self.wentry.grad.data.zero_()
            for layer in self.whidden:
                layer.grad.data.zero_()
            self.wout.grad.data.zero_()


    def _test_classic(self, test_data, test_data_label):
        accurrancy= 0
        test_data=torch.Tensor(test_data)
        test_data_label=torch.Tensor(test_data_label)
        x = Variable(test_data , requires_grad=False)
        x=self._addBiais(x)
        y = Variable(test_data_label, requires_grad=False)
        
        # Calcul de la matrice de prediction avec les poids modifies plus haut
        # 1 tableau de prediction de taille 10 par ligne du tableau de test
        #y_pred = x.mm(self.wentry).clamp(min=0)
        y_pred = self.unlinear(x.mm(self.wentry))
        y_pred = self._addBiais(y_pred)
        for layer in self.whidden:
            y_pred = self.unlinear(y_pred.mm(layer))
            y_pred=self._addBiais(y_pred)

        y_pred = self.unlinear(y_pred.mm(self.wout))

        for i in range(len(test_data)):
            d = y_pred[i,:]
            valuesx, indicesx = torch.max(d, 0)
            indices2 = numpy.argmax(test_data_label[i, :])
            indices1 =  indicesx.data.numpy()[0]
            #print("predicted %f label %f" % (indices1,indices2  ))
            if (indices1==indices2):
                accurrancy += 1

        if self.debug:
            print("Valeurs bien predit: %d " % (accurrancy))
            print("Valeurs mal predit:  %d " % (len(test_data) - accurrancy))
            print("Taux de reussite:    %f " % ((float(accurrancy)/len(test_data)) * 100))
        return (float(accurrancy)/len(test_data)) * 100

    def _test_optimize(self, test_data, test_data_label):
        accurrancy = 0
        test_data = torch.Tensor(test_data)
        test_data_label = torch.Tensor(test_data_label)
        x = Variable(test_data, requires_grad=False)
        y_pred = self.model(x)

        for i in range(len(test_data)):
            d = y_pred[i, :]
            valuesx, indicesx = torch.max(d, 0)
            indices2 = numpy.argmax(test_data_label[i, :])
            indices1 = indicesx.data.numpy()[0]
            # print("predicted %f label %f" % (indices1,indices2  ))
            if (indices1 == indices2):
                accurrancy += 1
        return (float(accurrancy) / len(test_data)) * 100

    def _addBiais(self, matrix):
        # Ajout du biais sur les nouvelles donnees
        b = Variable(torch.ones(len(matrix), 1), requires_grad=False)
        return torch.cat((matrix, b), 1)
