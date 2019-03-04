# -*- coding: utf-8 -*-
import scipy
from scipy.io import loadmat
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

# chargement de la base de donnée
mnist = loadmat("./mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

nbChiffre = [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]


def getAllTrainingDataOfLabel(label, ratio=0.8):
    quantite = int(ratio * nbChiffre[label])
    count = 0
    data = []
    indice = 0
    while count < quantite:
        if mnist_label[indice] == label:
            data.append(mnist_data[indice])
            count += 1
        indice += 1
    data = np.array(data)
    return data.transpose()


def apply_svd(label, basis_size):
    data = getAllTrainingDataOfLabel(label)
    print("Data loaded")
    data = np.array(data,dtype=np.float64)
    U, S, V = scipy.sparse.linalg.svds(data, k=basis_size)
    return U


def init_bases_SVD(M,nom_fichier_de_sauvegarde ="DATA_SVD.py" ):
    bases_SVD = []  # bases_SVD[n] := la base des u_i pour le chiffre n
    for i in range(10):
        bases_SVD.append(apply_svd(i, M))
    save_svd_data(bases_SVD,nom_fichier_de_sauvegarde)


def save_svd_data (base,nom_fichier_de_sauvegarde) :
    N = 784
    fic = open(nom_fichier_de_sauvegarde, "w")
    fic.write("#--coding: utf-8--\n")
    fic.write("bases_SVD = [")

    for k in range(10):
        fic.write("[")
        string = ""
        for i in range(N):
            string += "["
            lst = ','.join(str(e) for e in base[k][i])
            string += lst
            string += "]"
            if i < (N - 1):
                string += ","
        fic.write(string)
        fic.write("]")
        if k < 9:
            fic.write(",")
        fic.write("\n")
    fic.write("]\n")
    fic.close()


taille_de_la_base = 8
nom_fichier_sauvegarde = "DATA_"+str(taille_de_la_base)+"_SVD.py"
init_bases_SVD(nom_fichier_sauvegarde)
