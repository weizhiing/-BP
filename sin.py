import BP_net as bp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# todo: not softmax

network = bp.BP_Network([1, 20, 1], 0.1, 'sigmoidal')


input_x = np.random.uniform(-np.pi, np.pi, 1000)
output_y = np.sin(input_x)
plt.scatter(input_x, output_y)

# network.train(input_x, output_y, 100)
for loop in tqdm(range(500)):
    for i in range(len(input_x)):
        network.train_start(np.array([input_x[i]]), np.array([output_y[i]]), 1)


prediction = np.ndarray(len(input_x), np.object)
input_x = np.random.uniform(-np.pi, np.pi, 1000)
output_y = np.sin(input_x)
for i in range(len(input_x) // network.structure[0]):
    learning_data = np.ndarray(network.structure[0], np.object)
    for k in range(network.structure[0]):
        learning_data[k] = input_x[i * network.structure[0] + k]
    prediction[i] = network.forward_count(learning_data)
error = 0
for i in range(1000):
    error += abs(prediction[i] - output_y[i])

error = error / 1000
print('error: ' + str(error[0]))
plt.scatter(input_x, prediction)
plt.show()
