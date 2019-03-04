#--coding: utf-8--
import numpy as np
import load_DB as ldb
import DATA_SVD2 as DATA


"""
Entree: k, le nombre de vectuers voulus pour chaque base
Sorite: bases_k tel que bases_k[n] := les k premiers u_i pour le chiffre n
"""
def gen_U_k(k):
    bases_k = []
    for n in range(10):
        U_k = [DATA.bases_SVD[n][i] for i in range(k)]
        print(np.array(U_k).shape)
        bases_k.append(np.array(U_k))
    return bases_k


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

