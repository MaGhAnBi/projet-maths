#--coding: utf-8--
import numpy as np
import load_DB as ldb
import DATA_SVD as DATA
"""
Entrees: U := la bases des u_i d'un chiffre, image := l'indice d'un image non connu 
Sortie: la distance minimal de l'image au plan Vect(U)
"""
U_0 = np.array(DATA.bases_SVD[0])
print(U_0)
def distance_de_base(U,image):
    v = ldb.getData(image)
    v = np.array(v).reshape((784,1))
    M = np.eye(784)-np.matmul(U,U.transpose()) # cf rapport collectif
    return np.norm(np.matmul(M,v))
