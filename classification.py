import DATA_matrice_moyenne as DATA
import load_DB as ldb
import numpy as np
import SVD
import generateSVD
from scipy.spatial import distance
import GenerateTransformedData as generateT


def classificationTangeante(indice,db):
    print("Debut distance tangente")
    p=ldb.getData(indice)
    mini=np.inf
    derivs = ldb.getDerivationDB("translateX.mat")
    tp = np.array([derivs[indice]]).transpose()
    for i in range(len(db)):
        e = ldb.getData(db[i])
        te = np.array([derivs[db[i]]]).transpose()
        d=generateT.TangenteDistance(p,e,tp,te)
        if d<mini:
            index=i
            mini=d
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


def successRate( Test, algorithme,Training):
    label = []
    score = []
    matriceConfusion = np.zeros((10, 10), int)
    matriceConfusion_score = np.zeros((10, 10), int)

    N = len(Test)

    confused_positions = [[None]*N]*10


    confusion = 0
    for e in Test:
#        if i % 500 == 0:
#            print((i/14000)*100, "%")
#        i += 1
        label_e , score_e = algorithme(e)
        label.append(label_e)
        score.append(score_e)
    nbSuccess = 0
    for i in range(len(Test)):
        if label[i] == ldb.getLabel(Test[i]):
            nbSuccess += 1
        else:
            matriceConfusion[ldb.getLabel(Test[i]), label[i]] += 1
            if matriceConfusion_score[ldb.getLabel(Test[i]), label[i]] < score[i] :
                matriceConfusion_score[ldb.getLabel(Test[i]), label[i]] = score[i]
                confused_positions [ldb.getLabel(Test[i]), label[i]] = [i,Test[i]]
            confusion+=1
    matriceConfusion =matriceConfusion/confusion
    print("Confusions [ a , b , n ] : ", K_most_confused(matriceConfusion,matriceConfusion_score))
    return nbSuccess / N


#Training , Test = ldb.seperateData()
#Test_reduit= [Test[i] for i in range(10)]
#print(classificationTangeante(Test[0],Training),ldb.getLabel(Test[0]))
##print("classification tangente: ",successRate(Test_reduit,classificationTangeante,Training))
#print("Classification Moyenne :",successRate(Test,classificationMoyenne))
#print("Classification Moyenne :",successRate(Test,classificationMinkowski))
#
#print("Classification Cosinus :",successRate(Test,classificationChebyshev))
#print("Classification Cosinus :",successRate(Test,classificationManhattan))
#print("Classification Cosinus :",successRate(Test,classificationCosinus))

# GENERATION DONNEES :

#print("Classification SVD :", successRate(Test,classificationSVD))
