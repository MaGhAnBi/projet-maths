import load_DB as ldb
import numpy as np

Training , Test = ldb.seperateData()

"""
genere les N premirs vecteurs u_i de l'image du chiffre label
"""
def generate_svd_bases(label,Training,N):
    data = ldb.findChiffre_liste(label,Training) # trouves les indices correspondant au label
    A_c = np.zeros((784,len(data))) 
    # calcul de la matrice A_c (cf rapport collectif)
    for i in range(len(data)):
        vecteur_image = ldb.getData(data[i])
        for j in range(784):
            A_c[j,i] = vecteur_image[j]
            
    U,s,V = np.linalg.svd(A_c)
    U = U.transpose()#on cherche à extraire les vecteurs colonnes
    U_N = np.array([ np.array(U[i]) for i in range(N)])
    return U_N

bases_SVD = [] # bases_SVD[n] := la base des u_i pour le chiffre n  


def init_bases_SVD(N):
    for i in range(10):
        bases_SVD.append(generate_svd_bases(i,Training,N))
        
N = 10
init_bases_SVD(N)

fic = open("DATA_SVD.py","w")
fic.write("#--coding: utf-8--\n")
fic.write("bases_SVD = [")

for k in range(10):
    fic.write("[")
    string = ""
    for i in range(10):
        string+="["
        lst = ','.join(str(e) for e in bases_SVD[k][i])
        string+=lst
        string+="]"
        if i<9:
            string+=","
    fic.write(string)
    fic.write("]")
    if k<9:
        fic.write(",")
    fic.write("\n")
fic.write("]\n")
fic.close()   
