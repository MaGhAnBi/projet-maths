# -*- coding: utf-8 -*-
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# chargement de la base de donnée
mnist = loadmat("./mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]



nbChiffre = [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]

# retourne une liste des indices des chiffres n dans le tableau data
def findChiffre(n):
    lst = []
    for i in range(len(mnist_data)):
        if int(mnist_label[i])==n:
            lst.append(i)
    return lst

# retourne les pixel du chiffre de l'indice donné sous forme d'une liste
def getData(indice):
    if indice>=0 and indice<70000:
        return mnist_data[indice]

# affiche le chiffre de l'indice donné
def afficheChiffre(M):
    img = np.array(M)
    
    for i in range(len(img)):
        img[i] = 255-img[i]
        
    img = img.reshape((28,28))
    affichage = Image.fromarray(img,'L')
    affichage.show()

# retourne le label du chiffre à l'indice donné
def getLabel(indice):
    if indice>=0 and indice<70000:
        return int(mnist_label[indice])

# retourne un liste d'indices pour les donnés d'entrainement et une liste d'indice pour les donnés de test
def seperateData(ratio = 0.8):
    Training = []
    Test = []
    lst = [findChiffre(i) for i in range(10)]
    for n in range(10):
        limit = int(ratio*nbChiffre[n])
        for i in range(nbChiffre[n]):
            if i<limit:
                Training.append(lst[n][i])
            else:
                Test.append(lst[n][i])
    return Training, Test
