import DATA_matrice_moyenne as DATA
import load_DB as ldb
import numpy as np

def classificationMoyenne(indice):
    M = ldb.getData(indice)
    mini = np.inf
    index = 0
    for i in range(10):
        D = np.subtract(M,DATA.matrice_moyenne[i])
        d = np.linalg.norm(D)
       
        if d<mini:
            index = i
            mini = d
    return index

def successRate(Test):
    label = []
    for e in Test:
        label.append(classificationMoyenne(e))
    nbSuccess = 0
    for i in range(len(Test)):
        if label[i]==ldb.getLabel(Test[i]):
            nbSuccess+=1
    return nbSuccess/len(Test)

Training , Test = ldb.seperateData()

print(successRate(Test))