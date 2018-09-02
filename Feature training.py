
# coding: utf-8

# In[1]:


from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Flatten, Conv2D, MaxPooling1D, LSTM, RNN, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from sklearn.preprocessing import scale

import numpy as np
import cv2
import os
import re


# In[143]:


class Nural_network:
    def __init__(self, ):
#         Total Label available
        self.labels_class = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        
#         Number of class
        self.class_count = 62
    
#         No of sample in each class
        self.sample_count = 1016
    
#         Obsoleted; privously used in open cv
        self.sample_dimension = 128
    
#         Status flag
        self.isModel = False
        self.isTrained = False
        
#         Data
        self.train_data, self.train_labels = [],[]
        print("__init__")
    
    def getTrainingData(self, root, i):
#         read training file; feature data
        f = open(root+os.sep+"data_"+i, 'r')
        val = f.read()
        self.train_data = np.array(eval(val))
        f.close()
        
#         corresponding label 
        f = open(root+os.sep+"lable_"+i, 'r')
        lab = f.read()
        self.train_labels = np.array(eval(lab))
        f.close()
        
    def model(self):
#         TODO: Need to improve model
#         LSTM may give desired output in higher epoc
        if not self.isModel:
            self.isModel = True
            self.model = Sequential()
            self.model.add(Conv2D(128, (2, 2),  input_shape=(5, 2, 6), activation='relu'))
            self.model.add(Reshape(target_shape=((8, 64)), name='reshape'))
            self.model.add(LSTM(64, return_sequences=True))
            self.model.add(Flatten())
            self.model.add(Dense(self.class_count, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    def __shape(self, data):
#         Shape data in range -1 to 1
        for i in range(0, 6):
            data[:, :, i] = scale( data[:, :, i], axis=0, with_mean=True, with_std=True, copy=True )
        return np.reshape(data, (len(data),5,2,6))
    
    def Train(self, path='extracted'):
        self.isTrained = True
        for i in range(1, 127):
#             Total 64 file 1016/8 = 127
            print("Trainning Set : "+str(i))
    
#             Load the training data 
            self.getTrainingData(path, str(i))
            self.train_data = self.__shape(self.train_data)
            print(self.train_data.shape)
            
            self.model.fit(self.train_data, self.train_labels, epochs=100, batch_size=10)
        
    def load(self):
#         Load the saved model
        self.isTrained = True
        self.isModel = True
        self.model = load_model("feature.bin")
        
    def save(self):
#         Save the model
        self.model.save("feature.bin")
    
    def __key_func(self, x):
#         Function for sorting file in directory python "os"
        pat=re.compile("(\d+)\D*$")
        mat=pat.search(os.path.split(x)[-1]) # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1)) # right align to 10 digits.


# In[144]:


nn = Nural_network()


# In[145]:


# nn.load()
nn.model()


# In[146]:


nn.Train()


# In[ ]:


nn.save()


# In[138]:


print(nn.model.summary())

