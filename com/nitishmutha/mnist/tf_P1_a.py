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
EPOCH = 20000
LEARNING_RATE = 0.6

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

def NN_P1_A():
  x = tf.placeholder(tf.float32, [None, IMAGE_VECTOR])

  W = tf.Variable(tf.truncated_normal([IMAGE_VECTOR, OUTPUT_NODES],stddev=0.1))
  b = tf.Variable(tf.zeros([OUTPUT_NODES]))

  y = tf.nn.softmax(tf.matmul(x,W) + b)
  y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODES])

  saver = tf.train.Saver()

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  tf.summary.scalar("cross_entropy", cross_entropy)

  prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

  train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    y_true = np.argmax(mnist.test.labels, 1)
    y_p = tf.argmax(y, 1)

    if(TRAIN_MODE):
        print('---Running in train mode---')

        for i in range(EPOCH):
          batch_xs, batch_ys = mnist.train.next_batch(100)
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

          if i%200 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            accuracyList.append(train_accuracy)
            print("step %d, training accuracy %g" % (i, train_accuracy))

            errorList.append(cross_entropy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        #Test
        acc, pred = sess.run([accuracy, y_p], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test accuracy %f" % acc)

    else:
        # Test
        print('---Running in test mode---')

        saver.restore(sess,'trained_weights/P1_a/tf_P1_a.ckpt')
        acc, pred = sess.run([accuracy, y_p], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test accuracy %f" % acc)



    if (TRAIN_MODE):
        save_path = saver.save(sess,'trained_weights/P1_a/tf_P1_a.ckpt')
        print('Model weights saved in file: ', save_path)

        plotGraph(errorList, 'train cross entropy cost', 'epoch', 'cost', 'P1(a): Cross entropy cost')
        plotGraph(accuracyList, 'classification accuracy', 'epoch', 'accuracy', 'P1(a): Accuracy per epoch')

        print(confusion_matrix(y_true, pred))


if __name__ == '__main__':
    NN_P1_A()
