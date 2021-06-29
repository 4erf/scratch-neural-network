from neuron import Neuron
import random

class NeuralNetwork:
    def __init__(self, config, dropout, model = None, **kwargs):
        self.network = []
        self.dropout = dropout
        for i, n in enumerate(config[1:]):
            layer = []
            n_inputs = config[i]
            for j in range(n):
                w = None
                if model and len(model[i][j]) == n_inputs + 1:
                    w = model[i][j]
                else:
                    w = [random.uniform(-1.0, 1.0) for _ in range(n_inputs + 1)]
                layer.append(Neuron(**{
                    'initial_weights': w[:-1],
                    'initial_threshold': w[-1],
                    **kwargs
                }))
            self.network.append(layer)

    def predict(self, inputs):
        for layer in self.network:
            inputs = [neuron.predict(inputs) for neuron in layer]
        return inputs

    def train(self, old_inputs, targets):
        for i, layer in enumerate(reversed(self.network)):
            last_layer = i == 0
            first_layer = i == len(self.network) - 1
            for j, neuron in enumerate(layer):
                neuron.update(
                    self.get_input_layer(old_inputs) if first_layer else list(reversed(self.network))[i + 1],
                    None if last_layer else list(reversed(self.network))[i - 1],
                    j,
                    last_layer,
                    targets[j] if last_layer else None
                ) if random.random() >= self.dropout else None
        return

    def get_input_layer(self, inputs):
        return [Neuron(**{
            'initial_weights': [0],
            'initial_threshold': 0,
            'learning_rate': 0,
            'activation_fn': 'relu',
            'output': i
        }) for i in inputs]

    def get_model(self):
        return [[neuron.w + [neuron.t] for neuron in layer] for layer in self.network]
