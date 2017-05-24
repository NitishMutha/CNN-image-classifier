from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

tf.set_random_seed(123)

BATCH_SIZE = 100
IMAGE_VECTOR = 784
OUTPUT_NODES = 10
EPOCH = 2000
LEARNING_RATE = 0.001 #0.003

TRAIN_MODE = False  # Toggle this flag to train/test

accuracyList = []
errorList = []

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def convolution(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def plotGraph(data, label, xlabel, ylabel, title):
    plt.plot(data, label=label)
    plt.grid()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)
    plt.legend()
    plt.show()


def convolutionNN():
  x = tf.placeholder(tf.float32, [None, IMAGE_VECTOR])
  W_conv1 = weight_variable([3, 3, 1, 16])
  b_conv1 = bias_variable([16])

  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv1 = convolution(x_image, W_conv1) + b_conv1
  h_pool1 = maxpool(h_conv1)

  W_conv2 = weight_variable([3, 3, 16, 16])
  b_conv2 = bias_variable([16])

  h_conv2 = convolution(h_pool1, W_conv2) + b_conv2
  h_pool2 = maxpool(h_conv2)


  W_fc1 = weight_variable([7 * 7 * 16, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


  W_fc2 = weight_variable([1024, OUTPUT_NODES],)
  b_fc2 = bias_variable([OUTPUT_NODES])

  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

  y_expected = tf.placeholder(tf.float32,[None,OUTPUT_NODES])

  saver = tf.train.Saver()

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_expected))
  train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
  prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_expected,1))
  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    y_true = np.argmax(mnist.test.labels, 1)
    y_p = tf.argmax(y_conv, 1)

    if (TRAIN_MODE):
      print('---Running in train mode---')

      for i in range(EPOCH):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_expected: batch[1]})
        if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_expected: batch[1]})
          accuracyList.append(train_accuracy)
          print("step %d, training accuracy %g" %(i, train_accuracy))

          errorList.append(cross_entropy.eval(feed_dict={x: mnist.test.images, y_expected: mnist.test.labels}))

      acc, pred = sess.run([accuracy,y_p], feed_dict={x: mnist.test.images, y_expected: mnist.test.labels})
      print("test accuracy %f" %acc)

    else:
      #Test
      print('---Running in test mode---')
      saver.restore(sess, 'trained_weights/P1_d/tf_P1_d.ckpt')
      acc, pred = sess.run([accuracy, y_p], feed_dict={x: mnist.test.images, y_expected: mnist.test.labels})
      print("test accuracy %f" % acc)


    if (TRAIN_MODE):
      save_path = saver.save(sess, "trained_weights/P1_d/tf_P1_d.ckpt")
      print('Model weights saved in file: %s', save_path)

      plotGraph(errorList, 'train cross entropy cost', 'epoch', 'cost', 'P1(d): Cross entropy cost')
      plotGraph(accuracyList, 'classification accuracy', 'epoch', 'accuracy', 'P1(d): Accuracy per epoch')

      print(confusion_matrix(y_true, pred))

if __name__ == '__main__':
  convolutionNN()