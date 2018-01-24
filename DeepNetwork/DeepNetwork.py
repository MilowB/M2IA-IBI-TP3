import torch
from torch.autograd import Variable
import numpy

#AVANT DE COMMENCER VOICI LE LIEN VERS LA DOC:
# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


def deepNetwork(sizeEntries, sizeOut, sizeHidden, data, test_data, test_data_label):
    # A mon avis le truc qui ne va pas vient du paramétrage que l'on fait sur w1 et w2

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = len(data), sizeEntries, 5, sizeOut

    # Entrée
    x = Variable(data.type(dtype), requires_grad=False)
    # Prédiction
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Neurones et leurs poids (w1 couche d'entrée, w2 couche de sortie)
    w1 = Variable(torch.randn(D_in, N/50).uniform_(-0.1,0.1).type(dtype), requires_grad=True) # Couche entrée
    w2 = Variable(torch.randn(N/50, D_out).uniform_(-0.1,0.1).type(dtype), requires_grad=True) # Couche sortie

    #epsilon
    learning_rate = 1e-5

    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        loss = (y_pred - y).pow(2).sum()
        print(t, loss.data[0])
        loss.backward()

        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()


    # Traitement du test pour voir la performance du réseau
    accurrancy= 0

    x = Variable(test_data , requires_grad=False)
    y = Variable(test_data_label, requires_grad=False)

    # Calcul de la matrice de prédiction avec les poids modifiés plus haut
    # 1 tableau de prédiction de taille 10 par ligne du tableau de test
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    for i in range(len(test_data)):
        # Récupération de la prédiction de la ligne i du test
        d = y_pred[i,:]
        # Valeur max afin d'identifier la valeur prédite (indice où se trouve le 1)
        valuesx, indicesx = torch.max(d, 0)
        indices2 = numpy.argmax(test_data_label[i, :])
        indices1 =  indicesx.data.numpy()[0]
        #print("predicted %f label %f" % (indices1,indices2  ))
        if (indices1==indices2):
            accurrancy += 1

    print("Valeurs bien predit: %d " % (accurrancy))
    print("Valeurs mal predit:  %d " % (len(test_data) - accurrancy))
    print("Taux de reussite:    %f " % ((float(accurrancy)/len(test_data)) * 100))
    