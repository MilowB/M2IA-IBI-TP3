import torch
from torch.autograd import Variable
import numpy

#AVANT DE COMMENCER VOICI LE LIEN VERS LA DOC:
# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


def deepNetwork(sizeEntries, sizeOut, sizeHidden, data, data_label, test_data, test_data_label):
    # A mon avis le truc qui ne va pas vient du parametrage que l'on fait sur w1 et w2

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = sizeEntries, sizeHidden, sizeOut

    # Entree
    entries = Variable(data.type(dtype), requires_grad=False)
    # Prediction
    labels = Variable(data_label.type(dtype), requires_grad=False)
    #y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Neurones et leurs poids (w1 couche d entree, w2 couche de sortie)
    w1 = Variable(torch.randn(D_in, sizeHidden).type(dtype), requires_grad=True) # Couche entree
    w2 = Variable(torch.randn(sizeHidden, D_out).type(dtype), requires_grad=True) # Couche sortie

    #epsilon
    learning_rate = 5e-4

    for t in range(40000):
        x = Variable(data[t].type(dtype), requires_grad=False)
        y = Variable(data_label[t].type(dtype), requires_grad=False)
        #x = entries[t]
        #y= labels[t]

        x = x.unsqueeze(0)
        #x.mm scalair, clam fonction relu, mm scalaire sur la sortie
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # erreur
        loss = (y_pred - y).pow(2).sum()
        #print(t, loss.data[0])
        loss.backward()

        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()


    # Traitement du test pour voir la performance du reseau
    accurrancy= 0
    # Calcul de la matrice de prediction avec les poids modifies plus haut
    # 1 tableau de prediction de taille 10 par ligne du tableau de test

    for i in range(len(test_data)):
        # Recuperation de la prediction de la ligne i du test
        x = Variable(test_data[i].type(dtype), requires_grad=False)
        x = x.unsqueeze(0)
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Valeur max afin d identifier la valeur predite (indice ou se trouve le 1)
        valuesx, indicesx = y_pred.t().max(0)
        indices2 = numpy.argmax(test_data_label[i])
        indices1 =  indicesx.data[0]
        #print indices2, indices1
        #print("predicted %f label %f" % (indices1,indices2  ))
        if (indices1==indices2):
            accurrancy += 1

    print("Valeurs bien predit: %d " % (accurrancy))
    print("Valeurs mal predit:  %d " % (len(test_data) - accurrancy))
    print("Taux de reussite:    %f " % ((float(accurrancy)/len(test_data)) * 100))
    