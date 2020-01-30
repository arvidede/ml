import random
import numpy as np

from ml.utils import relu, relu_derivative

class NeuralNetwork(object):

    def __init__(self, size):
        # size: [input, hidden, ..., hidden, output]
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(size[:-1], size[1:])]


    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = relu(np.dot(w, a) + b)
        return a


    def SGD(self, train, epochs, mini_batch_size, eta, test=None):

        train = list(train)
        n = len(train)

        if test is not None:
            test = list(test)
            n_test = len(test)

        for j in range(epochs):
            random.shuffle(train)
            mini_batches = [
                train[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test is not None:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test),n_test));
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * relu_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def show(self):
        print(self.num_layers, self.size, self.weights, self.biases)
