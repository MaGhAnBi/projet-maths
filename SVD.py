#--coding: utf-8--
import numpy as np
import load_DB as ldb



"""
Entrees: U := la bases des u_i d'un chiffre, image := l'indice d'un image non connu 
Sortie: la distance minimal de l'image au plan Vect(U)
"""
def distance_de_base(U,image):
    v = ldb.getData(image)
    v = v.reshape((784,1))
    M = np.eye(784)-np.matmul(U,np.array(U).transpose()) # cf rapport collectif
    Mv = np.matmul(M,v) # multiplication de M*v
    return np.linalg.norm(Mv)

