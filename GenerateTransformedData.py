import numpy as np
from scipy.linalg import qr
from scipy.io import savemat
import load_DB as ldb
import PIL
import matplotlib.pyplot as plt
from random import randint


def translationDB(translate):
    """
    Fonction : calcul la derive de la translation translate pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """

    dic = {}
    nbData = 70000
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image)
        df_pos = translate(data,1) #s(image,1)
        df_neg = translate(data,-1) #s(image,-1)
        deriv = (df_pos-df_neg)/2 #s(image,1)-s(image,-1) approsximation de la dérivée de s au point (image,0)
        dic["derivation"][i] = deriv
        i+=1
    savemat(translate.__name__, dic,do_compression=True)


def translateX(image, alphaX):
    """
    translateX : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaX suivant l'Horizontal
    """
    image = image.reshape((28, 28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaX >= 0:
        imageTranslated[:, alphaX:image.shape[1]] = image[:, 0:(image.shape[1] - alphaX)]
    else:
        image = np.flip(image, (0, 1))
        alphaX = -alphaX
        imageTranslated[:, alphaX:image.shape[1]] = image[:, 0:(image.shape[1] - alphaX)]
        imageTranslated = np.flip(imageTranslated, (0, 1))
    return imageTranslated.reshape(784)


def translateY(image, alphaY):
    """
    translateY : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaY suivant la verticale
    """
    image = image.reshape((28, 28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaY >= 0:
        imageTranslated[alphaY:image.shape[1], :] = image[0:(image.shape[1] - alphaY), :]
    else:
        image = np.flip(image, (0, 1))
        alphaY = -alphaY
        imageTranslated[alphaY:image.shape[1], :] = image[0:(image.shape[1] - alphaY), :]
        imageTranslated = np.flip(imageTranslated, (0, 1))
    return imageTranslated.reshape(784)


def TangenteDistance(p, e, Tp, Te):
    """
    Fonction calcul la distance tangente entre deux images p,e
    Entrées: p,e: deux images,  Tp: la matrice des transormations de p, Te: la matrice des transformations des e
    Tp et Te ont la même dimension
    Sortie: d: la distance tangente de p et e
    """
    lp,cp=Tp.shape
    le,ce=Te.shape
    A = np.zeros((lp,cp+ce))
    A[:, 0:cp-1] = -1 * Tp[:, :]
    A[:,cp:ce+cp] = Te[:, :]
    Q, R = qr(A)  
    lr,cr=R.shape
    lq,cq=Q.shape
    Q2 = Q[:,cr:cq]
    b = p - e
    d = Q2.transpose()@b
    return np.linalg.norm(d)



Training, Test = ldb.seperateData()
alpha = 4
translationDB(translateX)
#ldb.resetDataBase("translateX.mat")
derivs = ldb.getDerivationDB("translateX.mat")
print(derivs[0])
# M = ldb.getData(10)
# trX = translateX(M,5)
# trY = translateY(M,5)
##translation(trainingData,0)
##img = translateX(M,1)
# plt.imshow(M.reshape(28,28))
# plt.show()
# plt.imshow(trX)
# plt.show()
# plt.imshow(trY)
# plt.show()
