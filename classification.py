import DATA_matrice_moyenne as DATA
import load_DB as ldb
import numpy as np
import SVD
import generateSVD
from scipy.spatial import distance
def classificationMoyenne(indice):
    M = ldb.getData(indice)
    mini = np.inf
    index = 0
    for i in range(10):
        D = np.subtract(M, DATA.matrice_moyenne[i])
        d = np.linalg.norm(D)

        if d < mini:
            index = i
            mini = d
    return index


def classificationChebyshev(indice):
    M=ldb.getData(indice)
    mini=np.inf
    index=0
    for i in range(10):
        d=distance.chebyshev(M,DATA.matrice_moyenne[i])
        if d<mini:
            index=i
            mini=d
    return index
"""
"""
def classificationMinkowski(indice):
    M=ldb.getData(indice)
    minimum=np.inf
    index=0
    for i in range(10):
        d=distance.minkowski(M,DATA.matrice_moyenne[i],3)
        if d<minimum:
            index=i
            minimum=d

    return index 
"""
"""
def classificationCorrelation(indice):
    M=ldb.getData(indice)
    minimum=np.inf
    index=0
    for i in range(10):
        d=distance.correlation(M,DATA.matrice_moyenne[i])
        if d<minimum:
            index=i
            minimum=d
    return index
"""

"""
def classificationManhattan(indice):
    M=ldb.getData(indice)
    mini=np.inf
    index=0
    for i in range(10):
        d=distance.cityblock(M,DATA.matrice_moyenne[i])
        if d<mini:
            index=i
            mini=d
    return index

def classificationCosinus(indice):
    M = ldb.getData(indice)
    normeM = np.linalg.norm(M)
    bestScore = 0
    index = 0
    for i in range(10):
        mean = DATA.matrice_moyenne[i]
        dotProduct = M.dot(mean)
        normeMean = np.linalg.norm(mean)
        cosinus = dotProduct / (normeM * normeMean)  # normeM et normeMean sont forcement differentes de 0
        if bestScore < cosinus:
            index = i
            bestScore = cosinus
    return index

"""
    Retourne la liste des K (K<=45) couples de chiffres les plus confondus ainsi que le nombre de fois qu'ils ont été confondus
"""

def K_most_confused(matrice, K=5):
    liste = []
    iu1 = np.triu_indices(matrice.shape[0])
    matrice = matrice + matrice.transpose()
    matrice[iu1] = -1
    for i in range(K):
        position = np.unravel_index(np.argmax(matrice, axis=None), matrice.shape)
        liste.append(np.append(np.flip(position, 0), matrice[position]).tolist())
        matrice[position] = -1
    return liste

k = 2
M = generateSVD.init_bases_SVD(k)

def classificationSVD(indice):
    scores = [0.]*10
    for label in range(10):
        scores[label] = SVD.distance_de_base(label, indice,M)

    return np.argmin(scores)


def successRate( Test, algorithme):
    label = []
    matriceConfusion = np.zeros((10, 10), int)
    i = 0
    N = len(Test)
    confusion = 0
    for e in Test:
#        if i % 500 == 0:
#            print((i/14000)*100, "%")
#        i += 1
        label.append(algorithme(e))
    nbSuccess = 0
    for i in range(len(Test)):
        if label[i] == ldb.getLabel(Test[i]):

            nbSuccess += 1
        else:
            matriceConfusion[ldb.getLabel(Test[i]), label[i]] += 1
            confusion+=1
    matriceConfusion =matriceConfusion/confusion
    print("Confusions [ a , b , n ] : ", K_most_confused(matriceConfusion))
    return nbSuccess / N

Training , Test = ldb.seperateData()

#print("Classification Moyenne :",successRate(Test,classificationMoyenne))
#print("Classification Moyenne :",successRate(Test,classificationMinkowski))
#
#print("Classification Cosinus :",successRate(Test,classificationChebyshev))
#print("Classification Cosinus :",successRate(Test,classificationManhattan))
#print("Classification Cosinus :",successRate(Test,classificationCosinus))

# GENERATION DONNEES :

print("Classification SVD :", successRate(Test,classificationSVD))
