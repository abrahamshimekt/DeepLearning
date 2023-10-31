import torch

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = torch.rand(n_neurons, n_inputs)
        self.bias = torch.rand(n_neurons)
        self.output = None

    def forward(self, inputs):
        self.output = torch.matmul(inputs, self.weights.t()) + self.bias
        return self.output


class NeuralNetwork:
    def __init__(self, n_features, n_hidden_neurons, n_classes):
        self.hidden_layers = []
        self.output_layer = DenseLayer(n_hidden_neurons[-1], n_classes)

        for i in range(len(n_hidden_neurons)):
            if i == 0:
                self.hidden_layers.append(DenseLayer(n_features, n_hidden_neurons[i]))
            else:
                self.hidden_layers.append(DenseLayer(n_hidden_neurons[i-1], n_hidden_neurons[i]))

    def forward(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = torch.relu(layer.forward(x))
        output = self.output_layer.forward(x)
        return output


# Define the network architecture
n_features = 5
n_hidden_neurons = [16, 16, 16]
n_classes = 5

# Create an instance of the neural network
network = NeuralNetwork(n_features, n_hidden_neurons, n_classes)

# Prepare input data
inputs = torch.randn(1, n_features)

# Perform forward pass
output = network.forward(inputs)

print(output)