#!/usr/bin/env python
# coding: utf-8

# In[5]:


# pip install opencv-python
# pip install opencv-contrib-python

from time import time
from numpy import uint8
from numpy.random import rand
import cv2
import os
import numpy as np


# In[9]:


class Extraction:
    """
        Frature extraction module  for extracting top 10 frature on an image 
    """
    def __init__(self,sample_folder='English Font Image', max_feture = 10, bundle_size = 127):
        """
            __init__(self,sample_folder='English Font Image' max_feture = 10, bundle_size = 127):
            sample_folder *string*: Base folder name for input images.
            max_feture *int*: Maximum number of feture to be extracted.
            bundle_size *int*: Define the number of sample in each extracted file.
        """
#         dimension of feature vactor is 300 ie. We can extrct 300 feture points maximum which is higher then 182 which is MAX 
        self.max_feture = max_feture
        self.bundle_size = bundle_size
        self.sample_folder = folsample_folderder
        
    def extract(self):
        """
            extract(): Extract feature of images located at ```English Font Image``` folder with directory structure
            *None*
            @return: None
            Input file structure
            English Font Image
            +-- Sample001
            +----- img001-00001.png
            +----- img001-00002.png
            +----- img001-00003.png
            +----- ...
            +----- img001-01016.png
            +-- Sample002
            +----- img001-00001.png
            +----- img001-00002.png
            +----- img001-00003.png
            +----- ...
            +----- img001-01016.png
            +-- Sample003
            +----- img001-00001.png
            +----- img001-00002.png
            +----- img001-00003.png
            +----- ...
            +----- img001-01016.png
            +-- ...
            +-- Sample00n
            +----- img001-00001.png
            +----- img001-00002.png
            +----- img001-00003.png
            +----- ...
            +----- img001-01016.png

            After running this script some file will created in **extracted** folder that contains the feature data in python array format

            The feature array contains *[point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave]*.
            In each file.
            Output file structure
            extracted
            +-- data1
            +-- data2
            +-- data3
            +-- ...
            +-- datan
            +-- lable1
            +-- lable2
            +-- lable3
            +-- ...
            +-- lablen
        """
        i = 0
        charecter = []
        lable = []
        for i in range(1, 1017):
#             processing each file 
#               i.e we are processing file such that We read first image in each class(62), do to second image in each class, then third image in each class 
            for j in range(1, 63):
#         Processing each folder
                folder = str(j);
                if(len(folder) == 1):
                    folder = "00" + folder
                if(len(folder) == 2):
                    folder = "0" + folder
                file = str(i)
                if(len(file) == 1):
                    file = "0000" + file
                if(len(file) == 2):
                    file = "000" + file
                if(len(file) == 3):
                    file = "00" + file
                if(len(file) == 4):
                    file = "0" + file
                    
                image = self.sample_folder+"\Sample"+ folder +"\img"+ folder +"-"+ file +".png"
#                 Path of image 1016*62 image 

                feture = self.__sift(image)
                keypoint = []
                f = 0;
                for point in feture:
#                     getting the values of keypoint
                    keypoint.append([point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave])
                    f += 1
            
                for cnt in range(f, self.max_feture):
#                     feeding 0 to rest of feature for making same size matrix i.e 20. (ref : self.max_feture); Keras takes same size input
                    keypoint.append([0, 0, 0, 0, 0, 0])
    
                keypoint.sort(key=lambda x:x[2])
        
                while len(keypoint) > self.max_feture:
                    keypoint.pop(0)

#                 appending the keypoint data to cherecter verible 
                charecter.append(keypoint[:])
#                 lable [62x1] matrix ; 1 for in class 0 for not
                lable_tmp = [0 for c in range(62)]
                lable_tmp[j-1] = 1
                lable.append(lable_tmp[:])
#             End of Image

            if(i % self.bundle_size == 0):
#                 For every 16 image data dump in "charecter" veriable will save in file 
#                 So that the matrix formed is  
#                 (992, 300, 7) in case of 16;
#                 (496, 300, 7) in case of 8
                self.__save(str(charecter), str(lable), str(int(i/self.bundle_size))) 
#                 Clear the memory
                charecter = []
                lable = []

    def __save(self, text, lable, lable_counter):
        """
            private function
            __save( text, lable, lable_counter):
            text *string*: Row feature data in python array(eval) format.
            lable *string*: Row lable data in python array(eval) format.
            lable_counter *int*: file name counter
            @return : None
        """
        f = open("extracted"+os.sep+"data_"+lable_counter, 'w')
        f.write(text)
        f.close()
        
        f = open("extracted"+os.sep+"lable_"+lable_counter, 'w')
        f.write(lable)
        f.close()
        
    def __sift(self, path):
        """
            private function
            __sift(path):
            path *string*: path of the image that you want to extract.
            @return : SIFT keypoint
        """
        img = cv2.imread(path)
        surf = cv2.xfeatures2d.SIFT_create()
        kp, des = surf.detectAndCompute(img,None)
        return kp
    


# In[3]:


# Create a instance of extraction module
e = Extraction() 
# Extract the feture 
e.extract()


# In[4]:


# Clean the veriable
del e


# In[ ]:





# In[ ]:




