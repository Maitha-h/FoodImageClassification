from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm
import pickle

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))
print(len(X_train), "TYPE: ", type(X_train[0]))
print(len(y_train))
X_train = X_train / 255.0

print(X_train.shape[1:])

'''
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
'''


print("building model.....")
model = Sequential()
print("first convolutional layer.........")
model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print("second convolutional layer.........")
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print("flattening layer.........")
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(units=512, activation='relu', kernel_constraint=max_norm(3)))

model.add(Dropout(rate=0.5))
model.add(Dense(units=3, activation='softmax'))
# difference: added dense layer instead of dropout(1) best to have last layer as softmaX
print("COMPILING .........")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # default learning rate is 0.001
print("FITTING THE MODEL.............")
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1)  # change size to 64 (also crashes my system) and try to run 128 crashes my system

model.save(filepath='3FOODs_Image_Classifier.h5')
print("DONE TRAINING AND FILE SAVED")

