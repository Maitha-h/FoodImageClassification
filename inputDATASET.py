import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pathlib
import pickle
import scipy.ndimage
import random

DATADIR = 'C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\images3'
data_dir = pathlib.Path(DATADIR)
Catagories = [item.name for item in data_dir.glob('*')]
print(Catagories)

IMG_SIZE = 256

''''
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
'''

training_data = []

'''
def create_training_data():
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                # add condition in this line  for creating testing dataset
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

'''

training_data = []
Dataset = []
testing_data = []


def create_dataset():
    i = 0
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                if i % 4 is 0:  # save 25% of data as testing data.
                    testing_data.append([new_array, class_num])
                else:
                    training_data.append([new_array, class_num])
            except Exception as e:
                pass
            i += 1

'''
def create_dataset_augmentation():
    i = 0
    j = 0
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                (rows, cols) = new_array.shape[:2]
                img2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-45, 45), 1)
                img2 = cv2.warpAffine(new_array, img2, (cols, rows))
                # img2 = imutils.rotate_bound(new_array, random.randint(1,45))
                img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
                if j == 0:
                    plt.imshow(new_array, cmap='gray')
                    plt.show()
                    plt.imshow(img2, cmap='gray')
                    plt.show()
                    j += 1
                Dataset.append([new_array, class_num])
                Dataset.append([img2, class_num])
            except Exception as e:
                pass
            i += 1

'''


def create_dataset_augmentation_random_rotation():
    i = 0
    j = 0
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img2 = scipy.ndimage.rotate(new_array, random.randint(-45, 45), mode='nearest')
                img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
                if j == 0:
                    plt.imshow(new_array, cmap='gray')
                    plt.show()
                    plt.imshow(img2, cmap='gray')
                    plt.show()
                    j += 1
                Dataset.append([new_array, class_num])
                Dataset.append([img2, class_num])
            except Exception as e:
                pass
            i += 1
            print("SAVING DATA NUMBER: ", i)
    random.shuffle(Dataset)


def create_dataset_augmentation_random_shift():
    i = 0
    j = 0
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img2 = scipy.ndimage.shift(new_array, random.randint(-25, 25), mode='mirror') #mirror is the best mode to keep the colors in place
                img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
                if j == 0:
                    plt.imshow(new_array, cmap='gray')
                    plt.show()
                    plt.imshow(img2)
                    plt.show()
                    j += 1
                Dataset.append([new_array, class_num])
                Dataset.append([img2, class_num])
            except Exception as e:
                pass
            i += 1
    random.shuffle(Dataset)


def separate_data():
    i = 0
    for i in range(len(Dataset)):
        if i % 10 is 0:  # save 25% of data as testing data.
            testing_data.append(Dataset[i])
        else:
            training_data.append(Dataset[i])


create_dataset_augmentation_random_rotation()
print("TRAINING DATA AFTER AUGMENTATION", len(Dataset),  "shape", type(Dataset[9]))
# create_dataset()
separate_data()
print("TRAINING DATA", len(training_data), "shape", type(training_data[9]))
print("TESTING DATA", len(testing_data),  "shape", type(testing_data[9]))

random.shuffle(training_data)
random.shuffle(testing_data)

# for sample in training_data[:10]:
#     print(sample[1])

X_train = []
y_train = []

for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)
print("Shape of my Training data", np.shape(np.array(X_train)))
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array(y_train).astype(int)

print("printing one sample of labels", y_train[2])

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

X_test = []
y_test = []

for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = np.array(y_test).astype(int)


pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

print("FINISHED IMPORTING DATA! ")

# run keras with regular image rotation to see if the accuracy is affected
# do shifting an run keras one more time
# add results to table and focus on tensorflow
