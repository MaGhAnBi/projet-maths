#--coding: utf-8--
import numpy as np
import load_DB as ldb
import DATA_SVD as DATA
"""
Entrees: U := la bases des u_i d'un chiffre, image := l'indice d'un image non connu 
Sortie: la distance minimal de l'image au plan Vect(U)
"""

def distance_de_base(U,image):
    v = ldb.getData(image)
    v = v.reshape((784,1))
    #print(np.matmul(U,np.array(U).transpose()).size)
    M = np.eye(784)-np.matmul(U,np.array(U).transpose()) # cf rapport collectif
    return np.linalg.norm(np.matmul(M,v))