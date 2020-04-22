# Reebbhaa Mehta 7042204305
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

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
    num_samples = actual.shape[0]
    log_like = -np.log(predict[np.arange(num_samples), actual.argmax(axis=1)])
    loss = np.sum(log_like) / num_samples
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
        self.update_hyper = 50
        self.learning_rate_update = .9
        self.min_learning_rate = 0.01

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

        train_samples = train_image_mat
        train_labels = train_label_mat

        num_batches = int(train_samples.shape[0] / self.batch_size)

        epoch_loss_list = []
        time_lr = 0
        for j in range(self.epoch):
            output = 0
            st = time.time()
            if (time.time() - time_lr) > 45:
                time_lr = time.time()
                train_samples, train_labels = shuffle_data(train_image_mat, train_label_mat)
                x = np.array_split(train_samples, num_batches, axis=0)
                y = np.array_split(train_labels, num_batches, axis=0)
                self.learning_rate = self.learning_rate*self.learning_rate_update if self.learning_rate*self.learning_rate_update > self.min_learning_rate else self.min_learning_rate
            
            accumulate_output = None

            for i in range(num_batches):
                output = self.forward(x[i])
                self.backward(output, y[i])
                
                if accumulate_output is None:
                    accumulate_output = np.copy(output)
                else:
                    accumulate_output = np.concatenate((accumulate_output, output), axis=0)
                
            l2 = self.regularization(train_samples.shape[0])
            epoch_loss = cross_entropy_loss(accumulate_output, train_labels, l2)
            epoch_loss_list.append(epoch_loss)
            # print("epoch number = {}, loss={:0.5f}, l2={:0.5f}, time={}".format(j, epoch_loss, l2, int(time.time()-st)))
            current_time = time.time() - START_TIME
            # if int(current_time/60) > 150:
            #     break
        predictions = self.test(train_image_mat)
        return epoch_loss_list, predictions 

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
        # print(solution.astype(int))
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

    train = np.array_split(train_image, 6, axis=0)
    label = np.array_split(train_label, 6, axis=0)
    pickle.dump((train[0], label[0]), open("dev_cut_train0.pkl", "wb"))
    pickle.dump((train[1], label[1]), open("dev_cut_train1.pkl", "wb"))
    pickle.dump((train[2], label[2]), open("dev_cut_train2.pkl", "wb"))
    pickle.dump((train[3], label[3]), open("dev_cut_train3.pkl", "wb"))
    pickle.dump((train[4], label[4]), open("dev_cut_train4.pkl", "wb"))
    pickle.dump((train[5], label[5]), open("dev_cut_train5.pkl", "wb"))



def load_dataset(return_test_label=False):
    
    test_image_raw = np.genfromtxt('test_image.csv', delimiter=',')
    test_image = normalize(test_image_raw)

    
    train_image_raw = np.genfromtxt('train_image.csv', delimiter=',')
    train_label_raw = np.genfromtxt('train_label.csv', delimiter=',')
    train_data, label_data = shuffle_data(normalize(train_image_raw), train_label_raw)
    
    if return_test_label:
        test_label_raw = np.genfromtxt('test_label.csv', delimiter=',')
        return train_data, label_data, test_image,  test_label_raw
    return train_data, label_data, test_image, None

def shuffle_split_test():

    np.random.seed(416)

    total_dataset = np.genfromtxt("total_dataset.csv", delimiter=",")
    total_labels = np.genfromtxt("total_labels.csv", delimiter=",")
    total_dataset = normalize(total_dataset)
    shuffled_dataset, shuffled_labels = shuffle_data(total_dataset, total_labels)
    max_num_sections = 8
    # img_list = np.array_split(shuffled_dataset, num_sections, axis=0)
    # label_list = np.array_split(shuffled_labels, num_sections, axis=0)
    for k in range(2, max_num_sections):

        img_list = np.array_split(shuffled_dataset, k, axis=0)
        label_list = np.array_split(shuffled_labels, k, axis=0)
        st = time.time()
        for i in range(k):

            train = img_list[i]
            label = label_list[i]
            test, test_label = None, None
            for j in range(k):
                if j == i:
                    continue
                else:
                    if test is None:
                        test = img_list[j]
                        test_label = label_list[j]
                    else:
                        test = np.concatenate((test, img_list[j]))
                        test_label = np.concatenate((test_label, label_list[j]))
            
            print("section number = {}, train.shape = {}, train_label.shape = {}, test.shape = {}, test_label.shape = {}".format(k, train.shape, label.shape, test.shape, test_label.shape))

            dataset_const = int(60000/train.shape[0])
            neural_network = NeuralNet(layer1=200, layer2=200, learning_rate=1., lambs=0.0006, regularize=True, batch_size=64, epoch=int(100*dataset_const*3), features=784, classes=10)

            label_hot = np.zeros((label.size, int(label.max()+1)))
            label_hot[np.arange(label.size), label.astype(int)] = 1

            losses, train_predictions = neural_network.train(train, label_hot)
            accuracy_train = neural_network.accuracy(train_predictions, label)

            test_predictions = neural_network.test(test)
            
            accuracy_test = neural_network.accuracy(test_predictions, test_label)
            
            print("test acc ", accuracy_test)
            print("train acc", accuracy_train)
            print("time", time.time() - st)
            st = time.time()
            plt.plot(losses)
            plt.title("train acc={:0.3f}, test acc={:0.3f},".format(accuracy_train, accuracy_test))
            plt.figure()
    
    plt.show()



def tune():
    START_TIME = time.time()
    return_test_label = True
    np.random.seed(416)
    train_60, label_60, test, test_label = load_dataset(return_test_label)
    lambd = 0.001

    for i in range(6):
        read_pkl = pickle.load(open("dev_cut_train{}.pkl".format(i), "rb"))
        train = normalize(read_pkl[0])
        label = read_pkl[1]
        dataset_const = int(60000/train.shape[0])
        neural_network = NeuralNet(layer1=128, layer2=128, learning_rate=0.7, lambs=lambd, regularize=True, batch_size=64, epoch=100*dataset_const, features=784,
                                   classes=10)
        lambd += 0.0002

        label_hot = np.zeros((label.size, int(label.max()+1)))
        label_hot[np.arange(label.size), label.astype(int)] = 1

        losses, train_predictions = neural_network.train(train, label_hot)
        accuracy_train = neural_network.accuracy(train_predictions, label)

        test_predictions = neural_network.test(test)
        if test_label is not None:
            accuracy_test = neural_network.accuracy(test_predictions, test_label)
            print("test acc ", accuracy_test)
            print("train acc", accuracy_train)
        plt.plot(losses)
        plt.title("lambda={:0.5f}, train acc={:0.3f}, test acc={:0.3f},".format(lambd, accuracy_train, accuracy_test))
        plt.figure()
    
    plt.show()

def actual_main():

    START_TIME = time.time()
    return_test_label = True
    np.random.seed(416)
    train, label, test, test_label = load_dataset(return_test_label)

    dataset_const = int(60000/train.shape[0])
    
    neural_network = NeuralNet(layer1=128, layer2=128, learning_rate=0.7, lambs=0.0015, regularize=True, batch_size=64, epoch=100*dataset_const, features=784,
                               classes=10)

    label_hot = np.zeros((label.size, int(label.max()+1)))
    label_hot[np.arange(label.size), label.astype(int)] = 1

    losses, train_predictions = neural_network.train(train, label_hot)
    accuracy_train = neural_network.accuracy(train_predictions, label)

    test_predictions = neural_network.test(test)
    
    if test_label is not None:
        accuracy_test = neural_network.accuracy(test_predictions, test_label)
        print("test acc ", accuracy_test)
        print("train acc", accuracy_train)
    
    plt.plot(losses)
    plt.title("train acc={:0.3f}, test acc={:0.3f},".format(accuracy_train, accuracy_test))
    # plt.figure()
    plt.show()

if __name__ == "__main__":
    START_TIME = time.time()
    shuffle_split_test()
    # split_dataset()
    # exit()
# split dataset into random sized datasets/ what if its an odd number. what if the batch size isnt divisible by the total number of samples. Run the neural network training and test it on those datasets. verify that the accuracy is above 92 in all cases. 
# reduce the learning rate every epoch by x%. 
# increase the batch size. 
    # default_args = ["train_image.csv" , "train_label.csv", "test_image.csv"]
    # if not len(sys.argv) > 1:
    #     load_dataset()