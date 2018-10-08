from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Flatten, Conv2D, MaxPooling1D, LSTM, RNN, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from sklearn.preprocessing import scale

D = Sequential()
depth = 64
dropout = 0.4
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (img_rows, img_cols, channel)
D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))
D.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))
D.add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))
D.add(Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
D.add(Dropout(dropout))
# Out: 1-dim probability
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()