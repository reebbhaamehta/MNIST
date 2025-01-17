# Reebbhaa Mehta 7042204305
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
import itertools


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return np.multiply(z, (1 - z))


def cross_entropy_derivative(predict, y_hot):
    num_samples = predict.shape[0]
    temp = predict - y_hot
    return temp/num_samples


def softmax(vector):
    shifted_vector = vector - np.max(vector, axis=1, keepdims=True)
    vector_exponent = np.exp(shifted_vector)
    return vector_exponent / np.sum(vector_exponent, axis=1, keepdims=True)


def cross_entropy_loss(predict, actual, l2):
    """
       p is the output from fully connected layer (num_examples x num_classes)
       y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    num_samples = actual.shape[0]
    log_likelihood = -np.log(predict[np.arange(num_samples), actual.argmax(axis=1)])
    loss = np.sum(log_likelihood) / num_samples
    loss_r = loss + l2
    return loss_r

def linear_back(dz, cache):
    a, w, b = cache

    m = a.shape[0]

    dw = 1. / m * np.dot(a.T, dz)
    db = 1. / m * np.sum(dz, axis=0)
    da = np.dot(dz, w.T)

    return da, dw, db.T


class NeuralNet:
    def __init__(self, layer1=10, layer2=10, learning_rate=0.2, lambs=0.1, regularize=False, batch_size=10, epoch=2, features=784, classes=10):
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
        self.bias1 = np.random.randn(self.nodes_layer1,)
        self.bias2 = np.random.randn(self.nodes_layer2,)
        self.bias_output = np.random.randn(self.classes,)
        self.cache = {}
        self.regularize = regularize
        self.lambs = lambs if self.regularize else 0

    def regularization(self, num_samples):
        return (np.sum(np.square(self.weights1)) + np.sum(np.square(self.weights2)) + np.sum(np.square(self.output_weights)))* self.lambs/(num_samples)


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
        output = softmax(y)
        self.cache_layers(x, a1, a2)
        return output

    def backward(self, output, y):
        num_samples = y.shape[0]
        layer3_params = self.cache[3]
        layer2_params = self.cache[2]
        layer1_params = self.cache[1]
        dl2 = 0 #(np.sum(self.weights1*2) + np.sum(self.weights2*2) + np.sum(self.output_weights*2))/y.shape[0]
        gradient = cross_entropy_derivative(output, y)
        da2, dw3, db3 = linear_back(gradient, layer3_params)
        dz2 = da2*sigmoid_derivative(layer3_params[0])
        da1, dw2, db2 = linear_back(dz2, layer2_params)
        dz1 = da1*sigmoid_derivative(layer2_params[0])
        da0, dw1, db1 = linear_back(dz1, layer1_params)

        reg_constant = (2*self.lambs/num_samples) if self.regularize else 0

        self.output_weights -= self.learning_rate * (dw3 + self.output_weights*reg_constant)
        self.bias_output -= self.learning_rate * db3

        self.weights2 -= self.learning_rate * (dw2 + self.weights2*reg_constant)
        self.bias2 -= self.learning_rate * db2

        self.weights1 -= self.learning_rate * (dw1 + self.weights1*reg_constant)
        self.bias1 -= self.learning_rate * db1

        return

    def train(self, train_image_mat, train_label_mat):
        validation_size = int(train_image_mat.shape[0]*0.2)
        validation_samples = train_image_mat[:validation_size, :]
        validation_samples_labels = train_label_mat[:validation_size, :]
        
        train_samples = train_image_mat[validation_size:, :]
        train_labels = train_label_mat[validation_size:, :]

        num_batches = int(train_samples.shape[0] / self.batch_size)

        x = np.array_split(train_samples, num_batches, axis=0)
        y = np.array_split(train_labels, num_batches, axis=0)
        loss_list, v_loss_list, v_accuracy_list = [], [], []
        for j in range(self.epoch):
            output = 0
            for i in range(num_batches):
                output = self.forward(x[i])
                l2 = self.regularization(x[i].shape[0])
                loss_list.append(cross_entropy_loss(output, y[i], l2))
                self.backward(output, y[i])
                v_loss, v_accuracy = self.validate(validation_samples, validation_samples_labels)
                v_accuracy_list.append(v_accuracy)
                v_loss_list.append(v_loss)
            print("epoch number = {}, loss={}, l2={}".format(j, loss_list[-1], l2))
            
        predictions = self.test(train_image_mat)
        return loss_list, predictions, v_loss_list, v_accuracy_list

    def validate(self, validation_data, validation_labels):
        y=self.forward(validation_data)
        y_index = np.argmax(y, axis=1)
        loss = cross_entropy_loss(y, validation_labels, self.regularization(y.shape[0]))

        y_accuracy = self.accuracy(y_index, validation_labels.argmax(axis=1))
        
        return loss, y_accuracy


    def test(self, test_data):
        x = test_data
        y = self.forward(x)
        y_index = np.argmax(y, axis=1)
        predictions = y_index
        # print(predictions)
        write_output(predictions)
        return predictions

    def accuracy(self, data, solution):
        # print(data)
        # print(solution)
        frac_correct = np.mean(data.astype(int) == solution.astype(int))
        return frac_correct*100


def read_input(arguments):
    train_image_matrix = np.genfromtxt(arguments.train_image, delimiter=',')
    test_image_matrix = np.genfromtxt(arguments.test_image, delimiter=',')
    train_label_matrix = np.genfromtxt(arguments.train_label, delimiter=',')
    test_label_matrix = np.genfromtxt('test_label.csv', delimiter=',')
    return train_image_matrix, train_label_matrix, test_image_matrix, test_label_matrix


def write_output(predictions):
    np.savetxt("test_predictions.csv", predictions.astype(int), fmt="%d")
    return

def shuffle_data(x,y):
  data = list(zip(x,y))
  np.random.shuffle(data)
  x_shuf, y_shuf = list(zip(*data))
  return np.array(x_shuf), np.array(y_shuf)

def normalize(x):
    return np.divide(x, 255)

def split_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_image', type=str, default='train_image.csv')
    parser.add_argument('train_label', type=str, default='train_label.csv')
    parser.add_argument('test_image', type=str, default='test_image.csv')
    args = parser.parse_args()

    train_image, train_label, test_image, test_label = read_input(args)

    train = np.array_split(train_image, 600, axis=0)
    label = np.array_split(train_label, 600, axis=0)
    pickle.dump((train[0], label[0]), open("dev_cut_train.pkl", "wb"))


def load_dataset(small_dataset=False, return_test_label=False):
    
    test_image_raw = np.genfromtxt('test_image.csv', delimiter=',')
    test_image = normalize(test_image_raw)

    if small_dataset:
        train_image_raw, train_label_raw = pickle.load(open("dev_cut_train.pkl", "rb"))
    else:
        train_image_raw = np.genfromtxt('train_image.csv', delimiter=',')
        train_label_raw = np.genfromtxt('train_label.csv', delimiter=',')
    train_data, label_data = shuffle_data(normalize(train_image_raw), train_label_raw)
    
    if return_test_label:
        test_label_raw = np.genfromtxt('test_label.csv', delimiter=',')
        return train_data, label_data, test_image,  test_label_raw
    return train_data, label_data, test_image, 0


# TODO: split dataset into train and test. After each epoch test on the test dataset if accuract > 90% stop training and start testing. 
if __name__ == "__main__":
    # split_dataset()
    # exit()
    small_dataset = False
    return_test_label = True
    np.random.seed(416)
    train, label, test, test_label = load_dataset(small_dataset, return_test_label)
    neural_network = NeuralNet(layer1=128, layer2=32, learning_rate=0.5, lambs=0.001, regularize=True, batch_size=5, epoch=100, features=784,
                               classes=10)
    label_hot = np.zeros((label.size, int(label.max()+1)))
    label_hot[np.arange(label.size), label.astype(int)] = 1

    losses, train_predictions, valid_loss, valid_accu  = neural_network.train(train, label_hot)
    accuracy_train = neural_network.accuracy(train_predictions, label)

    test_predictions = neural_network.test(test)
    accuracy_test = neural_network.accuracy(test_predictions, test_label)
    print(accuracy_test)

    # print(losses)
    plt.plot(losses)    
    plt.plot(valid_loss)

    plt.figure()
    plt.plot(valid_accu)
    plt.show()