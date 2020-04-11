# Reebbhaa Mehta 7042204305
import argparse
import pickle

import numpy as np
import scipy as sc
# import mnist_csv3
import matplotlib.pyplot as plt
import itertools


def sigmoid(z):
    # if x >= 0:
    #     z = np.exp(-x)
    #     return 1 / (1 + z)
    # else:
    #     # if x is less than zero then z will be small, denom can't be
    #     # zero because it's 1+z.
    #     z = np.exp(x)
    #     return z / (1 + z)
    return 1 / (1 + np.exp(-z))


def tanh(z):
    expz = np.exp(z)
    exp_z = np.exp(-z)
    return (expz - exp_z) / (expz + exp_z)


def sigmoid_derivative(z):
    return np.multiply(z, (1 - z))


def softmax_derivative(pi, pj, i, j):
    if i != j:
        return -np.multiply(pi, pj)
    else:
        return np.multiply(pi, (1-pj))


def cross_entropy_derivative(loss, output, y):
    """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
    m = y.shape[0]
    grad = output

    grad[range(m), y.astype(int)] -= 1
    grad = grad / m
    return grad


def tanh_derivative(z):
    return 1 - np.square(z)


def relu(z):
    return np.max(0, z)


def vectorize_input(train_image_matrix, test_image_matrix):
    train_image_transpose = np.transpose(train_image_matrix)
    test_image_transpose = np.transpose(test_image_matrix)
    return train_image_transpose, test_image_transpose


def softmax(vector):
    shifted_vector = vector - np.max(vector)
    vector_exponent = np.exp(shifted_vector)
    return vector_exponent / np.sum(vector_exponent)


def cross_entropy_loss(output, y):
    """
       p is the output from fully connected layer (num_examples x num_classes)
       y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(output)
    log_likelihood = -np.log(p[range(m), y.astype(int)])
    loss = np.sum(log_likelihood) / m
    return loss


def linear_back(dz, cache):
    a, w, b = cache
    # print("a w b -> ", a.shape, w.shape, b.shape)
    # print("dz", dz.shape)
    # print(a.T.shape)
    m = a.shape[1]
    dw = 1. / m * np.dot(a.T, dz)
    db = 1. / m * np.sum(dz, axis=1, keepdims=True)
    da = np.dot(dz, w.T)
    # print(da.shape, dw.shape, db.shape)
    return da, dw, db


class NeuralNet:
    def __init__(self, layer1=10, layer2=10, learning_rate=0.2, batch_size=10, epoch=2, features=784, classes=10):
        self.nodes_layer1 = layer1
        self.nodes_layer2 = layer2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.features = features
        self.classes = classes
        self.weights1 = np.random.randn(self.features, self.nodes_layer1)
        self.weights2 = np.random.randn(self.nodes_layer1, self.nodes_layer2)
        self.output_weights = np.random.randn(self.nodes_layer2, self.classes)
        self.Layers = 2
        self.bias1 = np.random.randn(self.batch_size, self.nodes_layer1)
        self.bias2 = np.random.randn(self.batch_size, self.nodes_layer2)
        self.bias_output = np.random.randn(self.batch_size, self.classes)
        # ch: a cache variable, a python dictionary that will hold some intermediate calculations that we will need
        # during the backward pass of the gradient descent algorithm.
        self.cache = {}
        self.gradient = {}
        self.loss = []

    def cache_layers(self, x, a1, a2):
        self.cache[1] = (x, self.weights1, self.bias1)
        self.cache[2] = (a1, self.weights2, self.bias2)
        self.cache[3] = (a2, self.output_weights, self.bias_output)

    def forward(self, x):
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = sigmoid(z2)
        y = np.dot(a2, self.output_weights) + self.bias_output
        output = y
        self.cache_layers(x, a1, a2)
        return output

    def backward(self, loss, output, y):
        layer3_params = self.cache[3]
        layer2_params = self.cache[2]
        layer1_params = self.cache[1]
        gradient = cross_entropy_derivative(loss, output, y)
        da2, dw3, db3 = linear_back(gradient, layer3_params)
        dz2 = sigmoid_derivative(da2)
        da1, dw2, db2 = linear_back(dz2, layer2_params)
        dz1 = sigmoid_derivative(da1)
        da0, dw1, db1 = linear_back(dz1, layer1_params)

        # print("db shape", db3.shape)
        # print("bias output shape", self.bias_output.shape)

        self.output_weights += -self.learning_rate * dw3
        self.bias_output += -self.learning_rate * db3

        self.weights2 += -self.learning_rate * dw2
        self.bias2 += -self.learning_rate * db2

        self.weights1 += -self.learning_rate * dw1
        self.bias1 += -self.learning_rate * db1

        return

    def train(self, train_image_mat, train_label_mat):
        x = np.array_split(train_image_mat, int(train_image_mat.shape[0] / self.batch_size), axis=0)
        y = np.array_split(train_label_mat, int(train_label_mat.shape[0] / self.batch_size), axis=0)
        loss_list = []
        for j in range(self.epoch):
            output = []
            for i in range(int(train_image_mat.shape[0] / self.batch_size)):
                output.append(self.forward(x[i]))
                loss = cross_entropy_loss(output[i], y[i])
                loss_list.append(loss)
                self.backward(loss, output[i], y[i])
        return loss_list

    def test(self, test_data):
        predictions = np.array(test_data.shape[0])
        x = np.array_split(test_data, int(test_data.shape[0] / self.batch_size), axis=0)
        for i in range(int(test_data.shape[0] / self.batch_size)):
            predictions.append(self.forward(x[i]))
        predictions = np.array(predictions)
        write_output(predictions)
        return


    # def save_network(self, file_name):
    #     pickle.dump(self, open(file_name, "wb"))


def read_input(arguments):
    train_image_matrix = np.genfromtxt(arguments.train_image, delimiter=',')
    test_image_matrix = np.genfromtxt(arguments.test_image, delimiter=',')
    train_label_matrix = np.genfromtxt(arguments.train_label, delimiter=',')
    test_label_matrix = np.genfromtxt('test_label.csv', delimiter=',')
    # train_image_matrix, test_image_matrix = vectorize_input(train_image_matrix, test_image_matrix)
    return train_image_matrix, train_label_matrix, test_image_matrix, test_label_matrix


def write_output(predictions):
    np.savetxt("test_predictions.csv", predictions, delimiter=",")
    return


def split_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_image', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('test_image', type=str)
    args = parser.parse_args()

    train_image, train_label, test_image, test_label = read_input(args)

    train = np.array_split(train_image, 6, axis=0)
    label = np.array_split(train_label, 6, axis=0)
    pickle.dump((train[0], label[0]), open("dev_cut_train.pkl", "wb"))


def load_dataset():
    train_image_matrix = np.genfromtxt('train_image.csv', delimiter=',')
    train_label_matrix = np.genfromtxt('train_label.csv', delimiter=',')

    test_image_matrix = np.genfromtxt('test_image.csv', delimiter=',')
    # test_label_matrix = np.genfromtxt('test_label.csv', delimiter=',')

    # train_data, label_data = pickle.load(open("dev_cut_train.pkl", "rb"))

    train_data = train_image_matrix
    label_data = train_label_matrix

    return train_data, label_data, test_image_matrix  # , test_label_matrix


if __name__ == "__main__":
    # split_dataset()
    # exit()

    train, label, test = load_dataset()
    neural_network = NeuralNet(layer1=64, layer2=32, learning_rate=0.01, batch_size=20, epoch=50, features=784,
                               classes=10)
    losses = neural_network.train(train, label)
    plt.plot(losses)
    plt.show()
    pickle.dump(neural_network, open("NN_11_5_20_20_2.pkl", "wb"))
    neural_network.test(test)

