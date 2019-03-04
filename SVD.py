#--coding: utf-8--
import numpy as np
import load_DB as ldb
import DATA_SVD as DATA


"""
Entree: k, le nombre de vectuers voulus pour chaque base
Sorite: bases_k tel que bases_k[n] := les k premiers u_i pour le chiffre n
"""
def gen_U_k(k):
    
    bases_k = []
    for n in range(10):
        U_k = [DATA.bases_SVD[n][i] for i in range(k)]
        bases_k.append(np.array(U_k).transpose())
    return bases_k

"""
Entree: bases_k
Sortie: une liste de matrice utilise pour le residuel des moindres carres
"""
def gen_M(bases_k):
    lst_M = []
    for n  in range(10):
        lst_M.append(np.eye(784)-np.matmul(bases_k[n],bases_k[n].transpose())) # cf rapport collectif
    return lst_M

"""
Entrees:  image := l'indice d'un image non connu , lst_M := une liste des matrice M 
Sortie: la distance minimal de l'image au plan Vect(U)
"""
def distance_de_base(label,image,lst_M):
    v = ldb.getData(image)
    v = v.reshape((784,1))
    M = lst_M[label]
    Mv = np.matmul(M,v) # multiplication de M*v
    return np.linalg.norm(Mv)

