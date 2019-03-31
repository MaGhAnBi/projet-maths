import numpy as np
from scipy.linalg import qr
from scipy.io import savemat
import load_DB as ldb
import PIL
import matplotlib.pyplot as plt
from random import randint


def translationDB(database,alpha,translate) :
    """
    Fonction : Translate de pas de -alpha à alpha en le sauvegardent le resultat dans translate.mat
    """
    dic = {}
    nbData = (2*alpha+1)*sum(ldb.nbChiffre)
    dic["data"] = np.zeros((nbData,784))
    dic["derivation"] = np.zeros((nbData,784))
    dic["label"] = np.zeros(nbData)
    i=0
    for image in database:
        img, T = translate(ldb.getData(image),alpha)
        dic["data"][i] = img
        dic["derivation"][i] = T
        dic["label"][i] = ldb.getLabel(image)
        i+=1
    savemat(translate.__name__,dic)



def translateX(image,alphaX) :
    """
    translateX : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaX suivant l'Horizontal
    """
    image = image.reshape((28,28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaX>= 0 :
        imageTranslated[:,alphaX:image.shape[1]] = image[:,0:(image.shape[1]-alphaX)] 
    else :
        image = np.flip(image,(0,1))
        alphaX = -alphaX
        imageTranslated[:,alphaX:image.shape[1]] = image[:,0:(image.shape[1]-alphaX)] 
        imageTranslated = np.flip(imageTranslated,(0,1)) 
    T = (imageTranslated-image)/alphaX
    return imageTranslated.reshape(784),T.reshape(784)

def translateY(image,alphaY) :
    
    """
    translateY : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaY suivant la verticale
    """
    image = image.reshape((28,28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaY>= 0 :
        imageTranslated[alphaY:image.shape[1],:] = image[0:(image.shape[1]-alphaY),:] 
    else :
        image = np.flip(image,(0,1))
        alphaY = -alphaY
        imageTranslated[alphaY:image.shape[1],:] = image[0:(image.shape[1]-alphaY),:] 
        imageTranslated = np.flip(imageTranslated,(0,1))
    T = (imageTranslated-image)/alphaY
    return imageTranslated.reshape(784), T.reshape(784)
    
def TangenteDistance(p,e,Tp,Te):
    """
    Fonction calcul la distance tangente entre deux images p,e
    Entrées: p,e: deux images,  Tp: la matrice des transormations de p, Te: la matrice des transformations des e
    Tp et Te ont la même dimension
    Sortie: d: la distance tangente de p et e
    """
    A = np.ones((1,2*784))
    A[:,0:Tp.shape[1]] = -1*Tp[:,:]
    A[:,Tp.shape[1]:2*Tp.shape[1]] = Te[:,:] # A = (-Tp Te)
    Q,R = qr(A) # decomposition QR de A
    Q2 = Q[:,R.shape[0]-1:Q.shape[1]]
    b = p-e
    d = Q2.transpose()@b
    return np.linalg.norm(d)

Training, Test = ldb.seperateData()    
translationDB(Training,1,translateX)
#M = ldb.getData(10)
#trX = translateX(M,5)
#trY = translateY(M,5)
##translation(trainingData,0)
##img = translateX(M,1)
#plt.imshow(M.reshape(28,28))
#plt.show()
#plt.imshow(trX)
#plt.show()
#plt.imshow(trY)
#plt.show()