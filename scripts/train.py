import os
import tensorflow as tf
import numpy as np
import keras
from cnnmodel import cnnModel
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

X_train = np.load('train1.npy')
Y_train = np.load('train_labels1.npy')
print('read over')
X_train = X_train.reshape(-1,28,28,3).astype('float32') / 255
#Y_train = to_categorical(Y_train.astype('float32'))
print(X_train.shape)
print(Y_train.shape)
#print(Y_train)
model = cnnModel((28,28,3))
model.compile(optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
                 loss='binary_crossentropy', metrics=['accuracy'])

# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)

# test_datagen = ImageDataGenerator(rescale = 1./255)

# training_set = train_datagen.flow_from_directory('train',
#                                                  target_size = (28, 28),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')

# test_set = test_datagen.flow_from_directory('test',
#                                             target_size = (28, 28),
#                                             batch_size = 32,
#                                             class_mode = 'binary')

# print('train.npy shape: ',X_train.shape)
# print('train_labels.npy shape: ', Y_train.shape)
model.fit(x=X_train, y=Y_train, batch_size=8, epochs=40)
print('fit over')
model.save('model.h5')

