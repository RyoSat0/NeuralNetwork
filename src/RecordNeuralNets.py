import mnist_loader
import dora
import json

train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
minidora = dora.Network([784,30,10])
minidora.SGD(train_data, 30, 10, 3.0, test_data)
minidora.SGD(train_data, 10, 10, 0.1, test_data)


def save_network(network, filename):
    network_data = {
        'biases': [b.tolist() for b in network.biases],  # List of arrays, each shape (y,1)
        'weights': [w.tolist() for w in network.weights]  # List of arrays, each shape (y,x)
    }

    with open(filename, 'w') as f:
        json.dump(network_data, f)

save_network(minidora, 'network_params.json')