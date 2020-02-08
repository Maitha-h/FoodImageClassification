import tensorflow as tf
import numpy as np
import glob
import pickle
from PIL import Image, ImageOps
import time
Start_time = time.time()
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow.compat.v1 as tf1


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
classes = 3

dropout_rate = 0.5
padding = 'SAME'

def conv2d(inputs, filters, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides= [1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.relu(out)

def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding=padding, strides=[1, stride_size, stride_size, 1])

def dense(inputs, weights):
    x = tf.nn.relu(tf.matmul(inputs, weights))
    return tf.nn.dropout(x, rate=dropout_rate)

def pcap(in_tensor, caps_dims):
    shape = in_tensor.get_shape().as_list()
    cap_count = shape[1]*shape[2]*shape[3]//caps_dims
    transposed = tf.transpose(in_tensor, [0,3,1,2])
    # print ("Transposed shape: ", tf.shape(transposed))
    return tf.reshape(transposed, [-1, 1, cap_count, caps_dims])


def hvc(in_tensor, out_caps, cap_dims, weight_decay=0.005):
    cap_size=in_tensor.get_shape().as_list()[2]
    # regularizer=weight_decay*tf.nn.l2_loss(tf.Variable(tf.truncated_normal([256 * 256 * 3, classes])))
    w_out_cap = tf.get_variable("w_out", shape=[out_caps, cap_size, cap_dims], initializer=tf.glorot_uniform_initializer())
    ocap = tf.reduce_sum(tf.multiply(in_tensor, w_out_cap), 2)
    return tf.nn.relu(ocap)


initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name):
    return tf.Variable(initializer(shape),  name=name, trainable=True)


output_classes = 3

shapes = [
    [3,3,3,32],
    [3,3,32,48],
    [3,3,48,64],
    [3,3,64,80],
    [3,3,80,96],
    [3,3,96,112],
    [3,3,112,128],
    [3,3,128,144],
    [3,3,144,160],
    [3,3,160,176],
    [3,3,176,192],
    [3,3,192,208],
]


weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format((i))))

def modelA(x, y):
    c = classes
    x = tf.cast(x, tf.float32)
    conv1 = conv2d(x, weights[0], stride_size=1)
    conv2 = conv2d(conv1, weights[1], stride_size=1)
    conv3 = conv2d(conv2, weights[2], stride_size=1)
    cap_dims = conv3.get_shape().as_list()[3] * conv3.get_shape().as_list()[3]
    pcap3 = pcap(conv3, cap_dims)
    ocap3 = hvc(pcap3, c, cap_dims, weight_decay=0.)
    logits3 = tf.reduce_sum(ocap3, axis=2, name="logits3")

    conv4 = conv2d(conv3, weights[3], stride_size=1)
    conv5 = conv2d(conv4, weights[4], stride_size=1)
    conv6 = conv2d(conv5, weights[5], stride_size=1)
    cap_dims = conv6.get_shape().as_list()[3] * conv6.get_shape().as_list()[3]
    pcap6 = pcap(conv6, cap_dims)
    ocap6 = hvc(pcap6, c, cap_dims, weight_decay=0.)
    logits6 = tf.reduce_sum(ocap6, axis=2, name="logits6")

    conv7 = conv2d(conv6, weights[6], stride_size=1)
    conv8 = conv2d(conv7, weights[7], stride_size=1)
    conv9 = conv2d(conv8, weights[8], stride_size=1)
    cap_dims = conv9.get_shape().as_list()[3] * conv9.get_shape().as_list()[3]
    pcap9 = pcap(conv9, cap_dims)
    ocap9 = hvc(pcap9, c, cap_dims, weight_decay=0.)
    logits9 = tf.reduce_sum(ocap9, axis=2, name="logits9")

    with tf.name_scope("logits"):
        logits = tf.stack([logits3, logits6, logits9], axis=2)
        logits = tf.layers.batch_normalization(logits, training=True)
        logits = tf.reduce_sum(logits, axis=2, name="logits")

    with tf.name_scope("loss"):
        preds = tf.nn.softmax(logits=logits)
        tf.stop_gradient(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y)) \
               + tf1.losses.get_regularization_loss()

    return preds, logits, loss


def model_single_layer(x):
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d(x, weights[0], stride_size=1)
    c1 = conv2d(c1, weights[1], stride_size=1)
    tf.shape(c1)
    # cap_dims = tf.reshape(tf.cast(c1.get_shape().as_list()[3] * c1.get_shape().as_list()[3], dtype=tf.int32), [])
    pcaps = pcap(c1, 256)
    ocap = hvc(pcaps, classes, 256)
    logits = tf.reduce_sum(ocap, axis=2)
    result = tf.nn.softmax(logits=logits)
    return result


def model(x):
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d(x, weights[0], stride_size=1)
    c1 = conv2d(c1, weights[1], stride_size=1)
    c1 = conv2d(c1, weights[2], stride_size=1)
    pcaps = pcap(c1, 512)
    ocap = hvc(pcaps, classes, 512)
    logits3 = tf.reduce_sum(ocap, axis=2)

    c2 = conv2d(c1, weights[3], stride_size=1)
    c2 = conv2d(c2, weights[4], stride_size=1)
    c2 = conv2d(c2, weights[5], stride_size=1)
    pcaps = pcap(c2, 256)
    ocap = hvc(pcaps, classes, 256)
    logits6 = tf.reduce_sum(ocap, axis=2)

    c3 = conv2d(c2, weights[6], stride_size=1)
    c3 = conv2d(c3, weights[7], stride_size=1)
    c3 = conv2d(c3, weights[8], stride_size=1)
    pcaps = pcap(c3, 128)
    ocap = hvc(pcaps, classes, 128)
    logits9 = tf.reduce_sum(ocap, axis=2)

    c4 = conv2d(c3, weights[9], stride_size=1)
    c4 = conv2d(c4, weights[10], stride_size=1)
    c4 = conv2d(c4, weights[11], stride_size=1)
    pcaps = pcap(c4, 64)
    ocap = hvc(pcaps, classes, 64)
    logits12 = tf.reduce_sum(ocap, axis=2)

    logits = tf.stack([logits3, logits6, logits9, logits12], axis=2)
    logits =  tf.layers.batch_normalization(logits, scale= True, training= True)
    logits = tf.reduce_sum(logits, axis=2, name="logits")
    result = tf.nn.softmax(logits=logits)
    return result

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.compat.v2.metrics.Mean()
train_accuracy = tf.compat.v2.metrics.SparseCategoricalAccuracy()
test_loss = tf.compat.v2.metrics.Mean()
test_accuracy = tf.compat.v2.metrics.SparseCategoricalAccuracy()
loss_object = tf.compat.v2.losses.SparseCategoricalCrossentropy()

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


num_epochs = 50

Train_loss =[]
Train_accuracy =[]
Test_loss = []
Test_accuracy =[]

def train_and_test(EPOCHS):
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

for i in range(2):
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
    Test_Accuracy = []
    Test_Loss = []
print("EXECUTION TIME: ", int((time.time() - Start_time) % 3600 // 60), ":",
      int((time.time() - Start_time) % 3600 % 60))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()