from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 100
IMAGE_VECTOR = 784
OUTPUT_NODES = 10
SOFTMAX = 'softmax'
RELU = 'relu'
STEP_SIZE = 0.01 #best 0.01
EPOCH = 10000

TRAIN_MODE = False  # Toggle this flag to train/test

accuracyList = []
errorList = []


def softmax(X, W, b):
    scores = np.dot(X,W) + b
    return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)


def softmax_derivative(hidden_layer, dscores, x, h):
    dW = np.dot(hidden_layer, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    return (dW, db)


def relu(X, W, b):
    return np.maximum(0, np.dot(X,W) + b)


def relu_derivative(dscores, W, X, hidden_layer):
    dhidden = np.dot(dscores, W)
    dhidden[hidden_layer <= 0] = 0
    #dhidden[hidden_layer > 0] = 1

    dW = np.dot(X, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    return (dW, db)


def computeLoss(prob, y):
    correct_logprob = -np.log(np.sum(prob*y,axis=1))
    return np.sum(correct_logprob)/BATCH_SIZE


def loadData():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)


def test(X_test, y_test, output_layer):

    scores = softmax(X_test, output_layer.W, output_layer.b)
    predicted_class = np.argmax(scores, axis=1)
    y_ = np.zeros((predicted_class.shape[0], 10))
    y_[np.arange(predicted_class.shape[0]), predicted_class] = 1
    return ((np.mean(y_ == y_test)),predicted_class)


def plotGraph(data, label, xlabel, ylabel, title):
    plt.plot(data, label=label)
    plt.grid()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)
    plt.legend()
    plt.show()

def oneHotToIndex(vec):
    return np.where(vec == 1)[1]


def plotConfusionMatrix(expected, predicted):
    cnf_matrix = confusion_matrix(oneHotToIndex(expected), predicted)
    print(cnf_matrix)


class NNLayer:

    def __init__(self, in_node, out_node, activation):
        self.W = 0.01 * np.random.randn(in_node, out_node)
        self.b = np.zeros((1, out_node))
        if activation == RELU:
            self.activation = relu
            self.activation_backp = relu_derivative
        elif activation == SOFTMAX:
            self.activation = softmax
            self.activation_backp = softmax_derivative

    def forwardProp(self, X):
        self.clayer = self.activation(X, self.W, self.b)
        return self.clayer

    def backPropPrams(self,in_node, out_node, X=None):
        return self.activation_backp(in_node, out_node, X, self.clayer)

    def updateParam(self, dW, db):
        self.W += -STEP_SIZE * dW
        self.b += -STEP_SIZE * db


def NeuralNet():

    #load input
    mnist = loadData()

    #configure neural net
    nn_outputNode = NNLayer(IMAGE_VECTOR,OUTPUT_NODES, SOFTMAX)

    if(TRAIN_MODE):
        #train and learn
        print('---Running in train mode---')
        for i in range(EPOCH):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

            outLayer_prob = nn_outputNode.forwardProp(batch_xs)

            loss = computeLoss(outLayer_prob, batch_ys)
            if i % 100 == 0:
                errorList.append(loss)
                print("iteration %d: loss %f" % (i, loss))

            #compute gradients
            dscores = outLayer_prob

            dscores -= batch_ys

            dscores /= BATCH_SIZE

            #backProp
            dW, db = nn_outputNode.backPropPrams(batch_xs.T, dscores)

            #update parameters
            nn_outputNode.updateParam(dW, db)
            if i % 100 == 0:
                accuracyList.append(test(batch_xs, batch_ys, nn_outputNode)[0])

        #test
        acc, predicted = test(mnist.test.images, mnist.test.labels, nn_outputNode)
        print('Testing accuracy: %.4f' % acc)

    else:
        #test mode
        print('---Running in test mode---')
        saved_var = np.load('trained_weights/P2_a/P2_a.npz')
        nn_outputNode.W = saved_var['dW']
        nn_outputNode.b = saved_var['db']

        acc, predicted = test(mnist.test.images, mnist.test.labels, nn_outputNode)
        print('Testing accuracy: %.4f' % acc)


    if(TRAIN_MODE):
        np.savez('trained_weights/P2_a/P2_a', dW=nn_outputNode.W, db=nn_outputNode.b)
        print('Model weights saved in file: trained_weights/P2_a/P2_a')

        plotGraph(errorList, 'train cross entropy cost', 'epoch', 'cost','P2(a): Decrease of cost over backprop iterations')
        plotGraph(accuracyList, 'classification accuracy', 'epoch', 'accuracy', 'P2(a): Accuracy per epoch')
        plotConfusionMatrix(mnist.test.labels, predicted)


if __name__ == '__main__':
    NeuralNet()


