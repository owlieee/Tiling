import pandas as pd
import cPickle as pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

myfile = open('../data/test_data.pickle', 'rb')
n = pickle.load(myfile)
myfile.close()

X = n['X']
y = n['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 30
X_test /= 30
Y_train = np_utils.to_categorical(y_train, 3)
Y_test = np_utils.to_categorical(y_test, 3)

model = Sequential()
model.add(ZeroPadding2D(padding = (4,4), data_format = 'channels_last',input_shape=(9,10,13)))
model.add(Convolution2D(32, 3, activation='relu', data_format='channels_last', input_shape=(9,10,13)))
model.add(Convolution2D(10, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)
