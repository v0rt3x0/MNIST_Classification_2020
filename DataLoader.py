#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install tensorflow
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


# In[5]:


def load_dataset():
    
 dataset = mnist.load_data()
 (trainX, trainY), (testX, testY) = dataset
 
 trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 testX = testX.reshape((testX.shape[0], 28, 28, 1))
 
 trainY = to_categorical(trainY)
 testY = to_categorical(testY)
    
 return trainX, trainY, testX, testY


# In[ ]:




