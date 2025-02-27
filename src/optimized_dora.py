# Optimized Version of Dora(Neural Network)
#
# I improved my neural network’s Stochastic Gradient Descent (SGD) by converting it into a fully matrix-based approach
# for backpropagation over a mini-batch. Instead of looping over individual training examples, the optimized version
# processes all examples in a mini-batch simultaneously using a matrix X = [x1, x2, ..., xn]
#
# It takes full advantage of the fact that NumPy is optimized for matrix operations and its efficiency in linear algebra routines.
# Specifically:
# -NumPy uses optimized BLAS (Basic Linear Algebra Subprograms) libraries, allowing entire vectors/matrices to be
#  processed in parallel instead of one element at a time, efficiently leveraging CPU instructions.
# -When using loops, data is fetched from scattered memory locations, causing cache misses.
#  A matrix-based approach keeps data together in memory, improving cache locality and reducing memory accesses.
#
# The speedup comes from SIMD parallelism and cache efficiency.
#
# This optimization reduced the training time by 74.03% 🚀
# Now my neural network learns significantly faster!
import random

import numpy as np

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        xs_list = [pair[0] for pair in mini_batch]
        ys_list = [pair[1] for pair in mini_batch]
        xs_matrix = np.column_stack(xs_list)
        ys_matrix = np.column_stack(ys_list)

        nabla_b, nabla_w = self.backprop(xs_matrix,ys_matrix)

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def backprop(self, x, y):
        nabla_b = [np.zeros((b.shape[0],x.shape[1])) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Calculate the Output Error
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Propagate errors
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        nabla_b = [np.sum(nb, axis=1).reshape(nb.shape[0],1) for nb in nabla_b]

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))