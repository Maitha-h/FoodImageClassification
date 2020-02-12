import tensorflow as tf
import numpy as np
import pickle
import time

Start_time = time.time()
import matplotlib.pyplot as plt
import warnings
tf.compat.v1.enable_eager_execution()
warnings.filterwarnings("ignore")


train_features = pickle.load(open("X_train.pickle", "rb"))
print("X", len(train_features))
train_labels = pickle.load(open("y_train.pickle", "rb"))
print("y", len(train_labels), "Shape", tf.shape(train_labels))
test_features = pickle.load(open("X_test.pickle", "rb"))
print("X", len(test_features))
test_labels = pickle.load(open("y_test.pickle", "rb"))
print("y", len(test_labels))
print("DATA LOADED")
test_features = test_features / 255.0

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.shuffle(1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
test_dataset = test_dataset.shuffle(1024).batch(batch_size)

relu_alpha = 0.2
dropout_rate = 0.0
padding = 'SAME'


def conv2d(inputs, filters, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides=[1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.relu(out)


def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding=padding,
                            strides=[1, stride_size, stride_size, 1])


def dense(inputs, weights):
    # x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=relu_alpha)
    x = tf.nn.relu(tf.matmul(inputs, weights))
    return tf.nn.dropout(x, rate=dropout_rate)


initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name, is_training):
    return tf.Variable(initializer(shape), name=name, trainable=True)


output_classes = 3
shapes = [
    [3, 3, 3, 16],
    [3, 3, 16, 16],
    [3, 3, 16, 32],
    [3, 3, 32, 32],
    [3, 3, 32, 64],
    [3, 3, 64, 64],
    [3, 3, 64, 128],
    [3, 3, 128, 128],
    [3, 3, 128, 256],
    [3, 3, 256, 256],
    [3, 3, 256, 512],
    [3, 3, 512, 512],
    [2048, 800],  # weight of flattening layer changes according to size of the image
    #  [3600, 2400],
    #  [2400, 1600],
    #  [512, 800],
    [800, 64],
    [64, output_classes]
]

shapes1 = [
    [3, 3, 3, 8],
    [3, 3, 8, 16],
    [3, 3, 16, 32],
    [3, 3, 32, 64],
    [3, 3, 64, 128],
    [3, 3, 128, 256],
    [3, 3, 256, 512],
    [3, 3, 512, 1024],
    [3, 3, 1024, 1024],
    [3, 3, 1024, 2048],
    [3, 3, 2048, 2048],
    [3, 3, 2048, 4096],
    [16384, 800],  # weight of flattening layer changes according to size of the image
    #[3600, 2400],
    #[2400, 1600],
    #[512, 800],
    [800, 64],
    [64, output_classes]
]

weights = []
#for i in range(len(shapes1)):
#    weights.append(get_weight(shapes1[i], 'weight{}'.format((i)), True))

def model(x):
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d(x, weights[0], stride_size=1)
    c1 = conv2d(c1, weights[1], stride_size=1)
    p1 = maxpool(c1, pool_size=2, stride_size=2)

    c2 = conv2d(p1, weights[2], stride_size=1)
    c2 = conv2d(c2, weights[3], stride_size=1)
    p2 = maxpool(c2, pool_size=2, stride_size=2)

    c3 = conv2d(p2, weights[4], stride_size=1)
    c3 = conv2d(c3, weights[5], stride_size=1)
    p3 = maxpool(c3, pool_size=2, stride_size=2)

    c4 = conv2d(p3, weights[6], stride_size=1)
    c4 = conv2d(c4, weights[7], stride_size=1)
    p4 = maxpool(c4, pool_size=2, stride_size=2)

    c5 = conv2d(p4, weights[8], stride_size=1)
    c5 = conv2d(c5, weights[9], stride_size=1)
    p5 = maxpool(c5, pool_size=2, stride_size=2)

    c6 = conv2d(p5, weights[10], stride_size=1)
    c6 = conv2d(c6, weights[11], stride_size=1)
    p6 = maxpool(c6, pool_size=2, stride_size=2)

    flatten = tf.reshape(p6, shape=(tf.shape(p6)[0], -1))

    # d1 = dense(flatten, weights[12])
    # d2 = dense(d1, weights[13])
    # d3 = dense(d2, weights[14])
    d4 = dense(flatten, weights[12])
    d5 = dense(d4, weights[13])
    logits = tf.matmul(d5, weights[14])

    return tf.nn.softmax(logits)


optimizer = tf.compat.v2.optimizers.SGD()
train_loss = tf.compat.v2.metrics.Mean()
train_accuracy = tf.compat.v2.metrics.SparseCategoricalAccuracy()
test_loss = tf.compat.v2.metrics.Mean()
test_accuracy = tf.compat.v2.metrics.SparseCategoricalAccuracy()
loss_object = tf.compat.v2.losses.SparseCategoricalCrossentropy()

def train_step(images, labels):
    global weights
    weights = []
    for i in range(len(shapes)):
        weights.append(get_weight(shapes[i], 'weight{}'.format(i), True))
    with tf.GradientTape() as tape:
        predictions = model(images, True)
        loss = loss_object(labels, predictions)
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(images, labels):
    global weights
    weights = []
    for i in range(len(shapes)):
        weights.append(get_weight(shapes[i], 'weight{}'.format(i), False))
    predictions = model(images, False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


num_epochs = 200

Train_loss =[]
Train_accuracy =[]
Test_loss = []
Test_accuracy =[]

def train_and_test(EPOCHS):
    weights = []
    for e in range(EPOCHS):
        print('Epoch {} out of {} {}'.format(e + 1, num_epochs, '--' * 20))
        for images, labels in train_dataset:
            train_step(tf.cast(images, tf.float32), labels)
        for test_images, test_labels in test_dataset:
            test_step(tf.cast(test_images, tf.float32), test_labels)

        print("Average Loss = {:.4f}".format(train_loss.result()))
        Train_loss.append(train_loss.result())
        print("Avrage Accuracy = {:.3f}%".format(train_accuracy.result() * 100))
        Train_accuracy.append(train_accuracy.result() * 100)
        print("Test Average Loss = {:.4f}".format(test_loss.result()))
        Test_loss.append(test_loss.result())
        print("Test Avrage Accuracy = {:.3f}%".format(test_accuracy.result() * 100))
        Test_accuracy.append( test_accuracy.result() * 100)

tf.executing_eagerly()

for i in range(10):
    print("TRIAL", (i + 1), "----------------------------------------------------------")
    train_and_test(num_epochs)
    plt.figure(1)
    plt.title('TRAIN ACCURACY')
    plt.plot(Train_accuracy)
    plt.figure(2)
    plt.title('TRAIN LOSS')
    plt.plot(Train_loss)
    plt.figure(3)
    plt.title('TEST ACCURACY')
    plt.plot(Test_accuracy)
    plt.figure(4)
    plt.title('TEST LOSS')
    plt.plot(Test_loss)
    print("Maximum Training Accuracy {:.3f}%".format((np.amax(Train_accuracy))))
    print("Maximum Testing Accuracy {:.3f}%".format((np.amax(Test_accuracy))))
    Train_accuracy = []
    Train_loss = []
    Test_accuracy = []
    Test_loss = []
print("EXECUTION TIME: ", int((time.time() - Start_time) // 3600), ":",
                          int((time.time() - Start_time) % 3600 // 60), ":",
                          int((time.time() - Start_time) % 3600 % 60))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
