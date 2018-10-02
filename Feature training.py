
# coding: utf-8

# In[ ]:


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
import random

# In[ ]:


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
        self.valid_data, self.valid_labels = [],[]
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

    def getValidData(self, root):
#         read training file; feature data
        f = open(root+os.sep+"data_1", 'r')
        val = f.read()
        self.valid_data = np.array(eval(val))
        f.close()
        
#         corresponding label 
        f = open(root+os.sep+"lable_1", 'r')
        lab = f.read()
        self.valid_labels = np.array(eval(lab))
        f.close()
        
    def model(self):
#         TODO: Need to improve model
#         LSTM may give desired output in higher epoc
        if not self.isModel:
            self.isModel = True
            self.model = Sequential()
            self.model.add(Conv2D(256, (2, 2),  input_shape=(5, 2, 6), activation='relu'))
            self.model.add(Reshape(target_shape=((8, 128)), name='reshape'))
            self.model.add(LSTM(128, return_sequences=True))
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
        sequence = [1, 2, 3, 4, 5, 6, 7, 8]
        random.shuffle(sequence)
        print('-----------')
        print(sequence)
        print('-----------')
        for i in sequence:
#             Total 64 file 1016/8 = 127
            print("Trainning Set : "+str(i))
    
#             Load the training data 
            self.getTrainingData(path, str(i))
            self.getValidData(path)
            
            self.train_data = self.__shape(self.train_data)
            self.valid_data = self.__shape(self.valid_data)
            print(self.train_data.shape)
            
            self.model.fit(self.train_data, self.train_labels, validation_data=(self.valid_data, self.valid_labels), epochs=1, batch_size=10)
            self.save()
        
    def load(self, path="feature.bin"):
#         Load the saved model
        self.isTrained = True
        self.isModel = True
        self.model = load_model(path)
        
    def save(self, path="feature.bin"):
#         Save the model
        self.model.save(path)
    
    def __key_func(self, x):
#         Function for sorting file in directory python "os"
        pat=re.compile("(\d+)\D*$")
        mat=pat.search(os.path.split(x)[-1]) # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1)) # right align to 10 digits.


# In[ ]:


nn = Nural_network()


# In[ ]:


nn.load()
# nn.model()


# In[ ]:

for i in range(10):
    print("Loop : "+str(i))
    nn.Train()


# In[ ]:


# nn.save()


# In[ ]:


print(nn.model.summary())

