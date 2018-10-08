# Optical-Character-Recognition 
A keras based module for trainning and Recognizing the synthetic charecter and then pridict the word using RCNN network
## Getting Started
The modeule can creatre RCNN model and it can train the model. using method of the call this modele can pridict the charecter in the image and then it makes word from cherecter after doing that it can mark all the word in image and produce a output again it create a folder containing name of that word in move the cropped word into it. size of moved image will be 64x64 for each word.

### Class Structure

Nural_network
- labels_class [veriable]
- class_count [veriable]
- sample_count [veriable]
- sample_dimension [veriable]
- isModel [veriable]
- isTrained [veriable]
- train_data [veriable]
- train_labels [veriable]
- __init __
- getTrainingData
- getValidData
- model
- __shape
- Train
- load
- save
- __key_func

### Prerequisites
For woring with this file you need to install the following library in system
- [Python](https://www.python.org)
- [numpy](http://www.numpy.org) 
  - pip install opencv-python
- [keras](https://keras.io)
  - pip install keras
- [Tensorflow](https://www.tensorflow.org)
  - pip install tensorflow
- [cv2](https://pypi.org/project/opencv-python). Visit [GitHub](https://github.com/skvark/opencv-python) 
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
***Create a model and train it:***
```
nn = Nural_network()
nn.model()
nn.Train()
nn.save()
nn.model.summary()
```
***Load the pre existing model:***
```
nn = Nural_network()
nn.load()
nn.Train()
nn.save()
nn.model.summary()
```