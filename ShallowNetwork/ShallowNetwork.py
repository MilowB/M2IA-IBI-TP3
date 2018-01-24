import torch
import math

class HiddenLayer:
    def __init__(self,entrySize, number, e):
        self.number = number
        self.neurons = [ self.Neuron(e,entrySize,i) for i in range(number) ]

    class Neuron:
        def __init__(self, e, size,index):
            self.e = e
            self.weight = torch.rand(size + 1) / 1000
            #self.weight = torch.add(self.weight, 0.01)
            self.index = index + 1

        def activity(self,data):
            self.y = 1.0 / ( 1.0 + math.exp(-torch.dot(data, self.weight)))
            return self.y

        def propagate(self, nextLayer):
            sum = 0.0
            for neuron1 in nextLayer.neurons:
                sum = sum + neuron1.error * neuron1.weight[self.index]
            self.error = self.y * ( 1.0 - self.y ) * sum

        def updateWeight(self, data):
            delta = self.e * data * self.error
            self.weight = self.weight - delta


class ExitLayer:
    def __init__(self, enterLayer,number,e):
        self.number = number
        self.previousLayer = enterLayer
        self.neurons = [ self.Neuron(e,len(enterLayer),i) for i in range(number)]

    class Neuron:
        def __init__(self, e, size, id):
            self.e = e
            self.weight = torch.zeros(size + 1)
            self.ti = torch.zeros(10)
            self.ti[id] = 1
            self.id = id
            self.error = 0

        def activity(self, data, index):
            act = torch.dot(self.weight, data)
            if index != -1:
                self.error = self.ti[index] - act
                self.error = self.error[0]
            return act

        def updateWeight(self, data):
            delta = self.e * data * self.error
            self.weight = self.weight - delta

class ShallowNetwork:
    def __init__(self, sizeEntries, sizeHidden, sizeExit, e):
        self.hiddenLayer = HiddenLayer(sizeEntries,sizeHidden,e)
        self.exitLayer = ExitLayer(self.hiddenLayer.neurons,sizeExit,e)

    def activity(self, data,label):
        # Ajout du biais
        data = torch.cat((torch.Tensor([1]), data), 0)
        index = (label == 1).nonzero()

        res = [1]
        #Calcul resultat couche intermediaire
        for neuron in self.hiddenLayer.neurons :
            res.append(neuron.activity(data))
        res = torch.FloatTensor(res)

        finalres =[]
        #Calcul resultat couche finale
        for neuron in self.exitLayer.neurons:
            act = neuron.activity(res, index)
            finalres.append(act)

        #Retro propagation et modification des poids de la couche du milieu
        for neuron in self.hiddenLayer.neurons:
            neuron.propagate(self.exitLayer)
            neuron.updateWeight(data)

        #Modification des poids de la couche finale
        for neuron in self.exitLayer.neurons:
            neuron.updateWeight(res)
        return finalres

    def predict(self,data):
        data = torch.cat((torch.Tensor([1]), data), 0)
        res=[1]
        for neuron in self.hiddenLayer.neurons:
            res.append(neuron.activity(data))
        res = torch.FloatTensor(res)

        val = None
        argmax = None

        for neuron in self.exitLayer.neurons:
            act = neuron.activity(res, -1)
            if val is None or act > val:
                val = act
                argmax = neuron
        return int((argmax.ti == 1).nonzero())
