import DATA_matrice_moyenne as DATA
import load_DB as ldb
import numpy as np
import SVD
import generateSVD
from scipy.spatial import distance
import GenerateTransformedData as generateT

def classificationTangeante(indice,testdb,db,derivsTest,derivsTraining):
    p=testdb[indice]
    tp = np.array([derivsTest[indice]]).transpose()
    lst = []
    count = 0
    for e in db:
        count+=1
        te = np.array([derivsTraining[e]]).transpose()
        lst.append(generateT.TangenteDistance(p,ldb.getData(e),tp,te))

   # print(generateT.TangenteDistance(p,ldb.getData(db[0]),tp,np.array([Te[db[0]]])))
    index = np.argmin(lst)
    return ldb.getLabel(db[index])


def classificationTangeanteY(indice,db):

    p=ldb.getData(indice)
    mini=np.inf
    print("Loading derivs")
    derivs = ldb.getDerivationDB("translateY.mat")
    tp = np.array([derivs[indice]]).transpose()
    print("Derivs loaded")
    for i in range(min(10,len(db))):
        e = ldb.getData(db[i])
        te = np.array([derivs[db[i]]]).transpose()
        d=generateT.TangenteDistance(p,e,tp,te)
        if d < mini:
            index=i
            mini=d
    return ldb.getLabel(db[index])


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
    return index, mini


def classificationChebyshev(indice):
    M=ldb.getData(indice)
    mini=np.inf
    index=0
    for i in range(10):
        d=distance.chebyshev(M,DATA.matrice_moyenne[i])
        if d<mini:
            index=i
            mini=d
    return index, mini
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

    return index , minimum
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
    return index, minimum
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
    return index, mini

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
    return index, bestScore

"""
    Retourne la liste des K (K<=45) couples de chiffres les plus confondus ainsi que le nombre de fois qu'ils ont ete confondus
"""

def K_most_confused(matrice,matriceConfusion_score, K=5):
    liste = []
    iu1 = np.triu_indices(matrice.shape[0])
    matrice = matrice + matrice.transpose()
    matrice[iu1] = -1
    confusion_to_show = []
    for i in range(K):
        position = np.unravel_index(np.argmax(matrice, axis=None), matrice.shape)
        liste.append(np.append(np.flip(position, 0), matrice[position]).tolist())
        matrice[position] = -1
        confusion_to_show.append(matriceConfusion_index[position])
    return liste


def p_root(value, p_value):
    root_value = 1 / float(p_value)
    return round (Decimal(value)**Decimal(root_value), 3)


def classificationNormeP(indice):
    p_value=1.5
    x=ldb.getData(indice)
    mini=np.inf
    index=0
    for i in range(10):
        y=DATA.matrice_moyenne[i]
        d=p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(x,y)), p_value)
        if d<mini:
            index=i
            mini=d
    return index, mini

k = 2
#M = generateSVD.init_bases_SVD(k)

def classificationSVD(indice):
    scores = [0.]*10
    for label in range(10):
        scores[label] = SVD.distance_de_base(label, indice,M)

    return np.argmin(scores), scores[np.argmin(scores)]


def successRate( Test,Training):
    nbSuccess = 0
    N = len(Test)
    derivs = ldb.getDerivationDB("translateX.mat")
    derivs = np.array(derivs)
    for e in Test:
        label_e  = classificationTangeante(e,Training,derivs)
        if label_e==ldb.getLabel(e):
            nbSuccess+=1
    return nbSuccess / N


#Training , Test = ldb.seperateData()
#derivs = ldb.getDerivationDB("translateX.mat")
#print(ldb.getLabel(classificationTangeante(Test[0],Training,derivs)),ldb.getLabel(Test[0]))
##print("classification tangente: ",successRate(Test_reduit,classificationTangeante,Training))
#print("Classification Moyenne :",successRate(Test,classificationMoyenne))
#print("Classification Moyenne :",successRate(Test,classificationMinkowski))
#
#print("Classification Cosinus :",successRate(Test,classificationChebyshev))
#print("Classification Cosinus :",successRate(Test,classificationManhattan))
#print("Classification Cosinus :",successRate(Test,classificationCosinus))

# GENERATION DONNEES :

#print("Classification SVD :", successRate(Test,classificationSVD))