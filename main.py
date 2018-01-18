import gzip
import cPickle
from Perceptron import *
from lecture_data import *

def main():
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    apprentissage = data[0]
    test = data[1]
    
    size = len(data[0][0][0])
    perceptron = Perceptron(size)

    total = 0
    gagne = 0
    for x in range(100000):
        d =  x % len(apprentissage[0])
        data = apprentissage[0][d]
        label = apprentissage[1][d]
        perceptron.activity(data, label)

    for x in range(len(test[0])):
        data = test[0][x]
        label = test[1][x]
        perceptron.activity(data, label)
        if int(label[perceptron.predict(data)]) == 1:
            gagne += 1
        total += 1
    print (float(gagne) / float(total)) * 100, "% (sur la base de test)"


if __name__ == '__main__':
    main()
