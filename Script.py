
# coding: utf-8

# In[51]:


from scipy.io import loadmat
mnist = loadmat("/Users/dicko/Documents/DICKO/L3/S6/3M101/Archives/mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

# A COMPLETER numberOfEachElementPRIME = [5922,7877,6990,7141,6824,6313,6876,7293,6825,6958] 



# In[13]:


print(mnist)


# In[49]:


print(mnist_label[5923])


# In[29]:


print(len(mnist_label))


# In[52]:


print(len(mnist_data[0]))

