import math

class Neuron:

    def __init__(self, initial_weights, initial_threshold, learning_rate, activation_fn, output = None, **kwargs):
        self.w = initial_weights[:]
        self.t = initial_threshold
        self.lr = learning_rate
        self.fn = self.relu if activation_fn == 'relu' else self.sigm
        self.dfn = self.drelu if activation_fn == 'relu' else self.dsigm
        self.a = output
        self.d = 0.0

    def update(self, prev_layer, next_layer, neuron_n, is_last_layer, target):
        self.d = self.last_layer_delta(target) if is_last_layer else self.intm_layer_delta(next_layer, neuron_n)
        self.w = [w + self.lr * self.d * prev_layer[j].a for j, w in enumerate(self.w)]
        self.t = self.t + self.lr * self.d
        return

    def intm_layer_delta(self, next_layer, neuron_n):
        return self.a * self.dfn(self.a) * sum(n.d * n.w[neuron_n] for n in next_layer)

    def last_layer_delta(self, target):
        return (target - self.a) * self.a * self.dfn(self.a)

    def calc_y(self, vec):
        """
            :param vec: Vector to evaluate
        """
        return sum(xi * wi for (xi, wi) in zip(vec, self.w)) + self.t

    def predict(self, ex):
        """
            :param ex: Example set (no d) [x1, x2, x3]
            :return output for the example (ReLU)
        """
        self.a = self.fn(self.calc_y(ex))
        return self.a

    def params(self):
        return {
            'weights': self.w,
            'threshold': self.t
        }

    def relu(self, o):
        return max(0, o)

    def sigm(self, o):
        return 1 / (1 + math.exp(-o))

    def drelu(self, y):
        return 0 if y == 0 else 1

    def dsigm(self, y):
        return (1 - y)

