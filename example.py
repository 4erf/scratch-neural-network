from neuralNetwork import NeuralNetwork

neural_network = NeuralNetwork(**{
    'learning_rate': 0.5,
    'config': [2, 2, 2],
    'activation_fn': 'sigm',
    'dropout': 0,
})

inputs = [0.05, 0.1]
out = neural_network.predict(inputs)
print(f'Output n1: {out[0]}')
print(f'Output n2: {out[1]}')

neural_network.train(inputs, [0.01, 0.99])

out = neural_network.predict(inputs)
print(f'Output n1: {out[0]}')
print(f'Output n2: {out[1]}')

# from saved model
model = neural_network.get_model()
neural_network = NeuralNetwork(**{
    'learning_rate': 0.5,
    'config': [2, 2, 2],
    'activation_fn': 'sigm',
    'dropout': 0,
    'model': model,
})

out = neural_network.predict(inputs)
print(f'Output n1: {out[0]}')
print(f'Output n2: {out[1]}')