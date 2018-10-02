# Open CV Feature Extraction
A class for extracting the image feature getting it in text file

## Getting Started

This module is used to extract the feture vector of an image. After extracting the data from image it will save at ```/extracted``` folder in same directory. It takes image of NxN size and extract SIFT feature using Open CV.
Our sample contains 128x128 size binary image. and extract 10(or N) frature in each image. The class select maximum top 10 SIFT frature of and image and rest of them will elenimate

### Class Structure
```
Extraction
-- self.max_feture = 10
-- __init__
-- extract
-- __save
-- __sift
```

### Prerequisites
For woring with this file you need to install the following library in system
- [Python](https://www.python.org/)
- [numpy](http://www.numpy.org/) 
  - pip install opencv-python
- [cv2](https://pypi.org/project/opencv-python/). Visit [GitHub](https://github.com/skvark/opencv-python) 
  - pip install opencv-contrib-python
- Cherecter dataset
   - given with project

### Operating System Support
- Windows
- Linux 

### Installing

**No need to install in system. Copy this file where you want to run**

Type ```pip install --``` if you are using native python installation and for anaconda distrubution use ```conda install --``` to install this pakage.

### Supported Python versions

- 3.5
- 3.6
- 3.7

### Running 

The following code will extract the feature of an image :

*Exaample 1*
```
e = Extraction()
e.extract()
del e
```
*Exaample 2*
```
e = Extraction()
e.extract(5) #top 5 feature
del e
```

### Data Sample
Test data at directory looks like this structure:

***Input file structure***
```
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
```
***Output file structure***
```
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
```