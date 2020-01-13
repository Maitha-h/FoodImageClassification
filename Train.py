from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time
NAME = "3 food Image Classifier - {}".format(int(time.time()))
tensorBoard = TensorBoard(log_dir='logs/{}'.format(NAME))

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))
print(len(X_train), "TYPE: ", type(X_train[0]))
print(len(y_train))
X_train = X_train / 255.0

print("IMG SIZE:",X_train.shape[1])

image_size = X_train.shape[1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

i = 1
# Model
model = Sequential()
# Adds layers in a sequential fashion one after the other
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 32, kernel_size=(3,3), input_shape=(image_size, image_size, 3),activation='relu', padding='valid', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 48, kernel_size=(3,3), input_shape=(32, 32, 3),activation='relu', padding='valid', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 64, kernel_size=(3,3), input_shape=(48, 48, 3),activation='relu', padding='valid', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 80, kernel_size=(3,3), input_shape=(64, 64, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 96, kernel_size=(3,3), input_shape=(80, 80, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 112, kernel_size=(3,3), input_shape=(96, 96, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 128, kernel_size=(3,3), input_shape=(112, 112, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 144, kernel_size=(3,3), input_shape=(128, 128, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
print("Adding layer: ", i)
i += 1
model.add(Conv2D(filters= 160, kernel_size=(3,3), input_shape=(144, 144, 3),activation='relu', padding='same', kernel_constraint= max_norm(3)))
model.add(Flatten())
# converts the output into a 1D array
model.add(Dense(units=512, activation='relu', kernel_constraint=max_norm(3)))
# creates actual prediction network
# The higher the number of units the greater the accuracy
# However, the higher the number of units the longer it takes to train
model.add(Dropout(rate=0.5))
# Drops out half the units used to increase reliability
model.add(Dense(units=3, activation='softmax'))
# number of units is the number of classes
# used to produce output for each of he 10 categories
print("Compiling")
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Into Fit Loop")
for i in range(1):
  TBLOGDIR="logs/trail_{}".format(i+1)
  model.fit(x=X_train, y=y_train, epochs=1, batch_size=16, validation_split=0.1, callbacks=[TensorBoard(log_dir=TBLOGDIR)])
  # print("Trail:", i+1, "Training Accuracy:", np.mean(hist.history['acc']))
# Started training the model