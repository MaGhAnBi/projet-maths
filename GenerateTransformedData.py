import numpy as np
import load_DB as ldb
import PIL
import matplotlib.pyplot as plt

"""
    Fonction : Translate d'un pas de 'alpha' à toutes les images de la base de données database
               en appliquant la translation 'translate'
"""

def translation (database,alpha,translate) :
    for image in database:
        translated = translate(ldb.getData(image),alpha)

"""
    translateX : (image : 1x784, alphaX : int  ) ==> imageTranslated : 28x28
    Fonction : Translate l'image 'image' de alphaX suivant l'Horizontal
"""

def translateX(image,alphaX) :
    image = image.reshape((28,28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaX>= 0 :
        imageTranslated[:,alphaX:image.shape[1]] = image[:,0:(image.shape[1]-alphaX)] 
    else :
        image = np.flip(image,(0,1))
        alphaX = -alphaX
        imageTranslated[:,alphaX:image.shape[1]] = image[:,0:(image.shape[1]-alphaX)] 
        imageTranslated = np.flip(imageTranslated,(0,1))
    return imageTranslated

"""
    translateY : (image : 1x784, alphaX : int  ) ==> imageTranslated : 28x28
    Fonction : Translate l'image 'image' de alphaY suivant la verticale
"""
def translateY(image,alphaY) :
    image = image.reshape((28,28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaY>= 0 :
        imageTranslated[alphaY:image.shape[1],:] = image[0:(image.shape[1]-alphaY),:] 
    else :
        image = np.flip(image,(0,1))
        alphaY = -alphaY
        imageTranslated[alphaY:image.shape[1],:] = image[0:(image.shape[1]-alphaY),:] 
        imageTranslated = np.flip(imageTranslated,(0,1))
    return imageTranslated
    
def TangenteClassificationDistance(dataBase,p,translation, p_translation_parameters, e_translation_parameters):
    
    indice = 0
    dmin = np.inf
    for e in dataBase :
        for alpha in p_translation_parameters:
                tp = translation(p,alpha) - p.reshape((28,28))
                for beta in e_translation_parameters:
                    e_matrix = ldb.getData(dataBase)
                    te = translation(e_matrix,beta) - e_matrix.reshape((28,28))
                    d = np.linealg.norm(p+tp*alpha - e_matrix - (te*beta))
                    if d < dmin : 
                        d = dmin
                        indice = e
                        
    
trainingData,testData = ldb.seperateData()
alpha = 10
translation(trainingData,alpha,translateX)
"""
M = ldb.getData(10)
trX = translateX(M,-5)
trY = translateY(M,-0)
#translation(trainingData,0)
#img = translateX(M,1)
plt.imshow(M.reshape(28,28))
plt.show()
plt.imshow(trX)
plt.show()
plt.imshow(trY)
plt.show()
"""