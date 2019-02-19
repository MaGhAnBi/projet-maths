# -*- coding: utf-8 -*-
import DATA_matrice_moyenne as DATA1
import DATA_matrice_variance as DATA2
import load_DB as ldb
import numpy as np

def classificationMoyenne(indice):
    M = ldb.getData(indice)
    mini = np.inf
    index = 0
    for i in range(10):
        IntervalleConfiance = np.add(DATA1.matrice_moyenne,DATA2.matrice_variance)
        D = np.subtract(M,IntervalleConfiance[i])
        d = np.linalg.norm(D)
       
        if d<mini:
            index = i
            mini = d
    return index

#
#def classificationCosinus(indice):
#    M = ldb.getData(indice)
#    normeM =  np.linalg.norm(M)
#    bestScore = 0 
#    index = 0
#    for i in range(10):
#        mean = DATA.matrice_moyenne[i] 
#        dotProduct = M.dot(mean)
#        normeMean =  np.linalg.norm(mean)
#        cosinus = dotProduct/(normeM*normeMean) # normeM et normeMean sont forcement differentes de 0
#        if bestScore<cosinus:
#            index = i
#            bestScore = cosinus
#            
#    return index

def successRate(Test,algorithme):
    label = []
    for e in Test:
        label.append(algorithme(e))
    nbSuccess = 0
    for i in range(len(Test)):
        if label[i]==ldb.getLabel(Test[i]):
            nbSuccess+=1
    return nbSuccess/len(Test)

Training , Test = ldb.seperateData()

print("Classification Moyenne :",successRate(Training,classificationMoyenne))
#
#print("Classification Cosinus :",successRate(Test,classificationCosinus))
