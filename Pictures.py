
# coding: utf-8

# In[123]:


from PIL import Image
import numpy as np
import random
import os

"""
Function find(label,address): 
    return a random picture labeled 'label' assuming images at address 'address' 
"""
def find(label=0,address="./pictures/"):
    numberOfEachElement = [6903,7877,6990,7141,6824,6313,6876,7293,6825,6958] ; 

    a = 1 ;
    b = numberOfEachElement[label] ; 
    choice = random.randint(a,b) ;
    fileName = str(address)+""+str(label)+"_"+str(choice)+".jpg" ;
    image = Image.open(fileName, 'r')
    image.show(np.asarray(image))
    image.close()
    #comment the above line before deleting '#' from bellow line. 
    #return image
    
    
"""
    Generate training and testing data. 
    @Parameters : 
    Ratio ∈ [0,1] : Training data ratio
    label : {0,1,...,9}
    
"""
def splitData(ratio=0.8,label=0,address="./pictures/",trainingPicturesDestination="./trainingPictures/",testingPicturesDestination="./testingPictures/"):
    """
        Creating directories if not exist
    """
    if not os.path.exists(trainingPicturesDestination):
        os.makedirs(trainingPicturesDestination)
    if not os.path.exists(testingPicturesDestination):
        os.makedirs(testingPicturesDestination)
        
    numberOfEachElement = [6903,7877,6990,7141,6824,6313,6876,7293,6825,6958] ; 
    
    a = 1 ;
    b = numberOfEachElement[label] ;
    currentLength = b ; 
    totalNumber = int(b*ratio)
    possibilities = np.arange(a, b+1, 1) ;
    """
        Training dataset
    """
    for i in range(totalNumber):  
        choice = random.randint(0,currentLength-1) ;
        fileName = (str(address)+str(label)+"_"+str(possibilities[choice])+".jpg") ;
        currentLength = currentLength - 1 ;
        possibilities[choice] = possibilities[currentLength] ;
        image = Image.open(fileName, 'r') ;
        image.save(trainingPicturesDestination+str(label)+"_"+str((i+1))+".jpg") ;
        image.close() ;
    
    """
        Testing dataset
    """
    for j in range(currentLength):  
        fileName = (str(address)+str(label)+"_"+str(possibilities[j])+".jpg") ;
        image = Image.open(fileName, 'r') ;
        image.save(testingPicturesDestination+str(label)+"_"+str((j+1))+".jpg")
        image.close() ;
        
    print("Done,",totalNumber,"pictures for training",b-totalNumber,"pictures for testing for label",label) ;


# In[122]:


find(address = "/Users/dicko/Documents/DICKO/L3/S6/3M101/caracteres/pictures/") ;
splitData(ratio=0.8,label=1) ; 

