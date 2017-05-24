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

HIDDEN_LAYER_1 = 256
HIDDEN_LAYER_2 = 256

BATCH_SIZE = 100
IMAGE_VECTOR = 784
OUTPUT_NODES = 10
EPOCH = 10000
LEARNING_RATE = 0.95

TRAIN_MODE = False  # Toggle this flag to train/test

accuracyList = []
errorList = []


def plotGraph(data, label, xlabel, ylabel, title):
    plt.plot(data, label=label)
    plt.grid()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)
    plt.legend()
    plt.show()


def NN_P1_C():

  x = tf.placeholder(tf.float32, [None, IMAGE_VECTOR])

  W1 = tf.Variable(tf.truncated_normal([IMAGE_VECTOR, HIDDEN_LAYER_1],stddev=0.1))
  b1 = tf.Variable(tf.zeros([HIDDEN_LAYER_1]))

  W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2],stddev=0.1))
  b2 = tf.Variable(tf.zeros([HIDDEN_LAYER_2]))

  W3 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_2, OUTPUT_NODES],stddev=0.1))
  b3 = tf.Variable(tf.zeros([OUTPUT_NODES]))

  h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

  h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

  y = tf.nn.softmax(tf.matmul(h2,W3) + b3)
  y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODES])

  saver = tf.train.Saver()

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

  prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    y_true = np.argmax(mnist.test.labels, 1)
    y_p = tf.argmax(y, 1)

    if (TRAIN_MODE):
      print('---Running in train mode---')

      for i in range(EPOCH):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
          accuracyList.append(train_accuracy)
          print("step %d, training accuracy %g" % (i, train_accuracy))

          errorList.append(cross_entropy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      #test
      acc, pred = sess.run([accuracy, y_p], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print("Test accuracy %f" % acc)

    else:
      print('---Running in test mode---')
      #Test
      saver.restore(sess, 'trained_weights/P1_c/tf_P1_c.ckpt')
      acc, pred = sess.run([accuracy, y_p], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print("Test accuracy %f" % acc)


    if (TRAIN_MODE):
      save_path = saver.save(sess, 'trained_weights/P1_c/tf_P1_c.ckpt')
      print('Model weights saved in file: ', save_path)

      plotGraph(errorList, 'train cross entropy cost', 'epoch', 'cost', 'P1(c): Cross entropy cost')
      plotGraph(accuracyList, 'classification accuracy', 'epoch', 'accuracy', 'P1(c): Accuracy per epoch')

      print(confusion_matrix(y_true, pred))

if __name__ == '__main__':
    NN_P1_C()