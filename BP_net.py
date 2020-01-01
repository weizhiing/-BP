import numpy as np


class BP_Network:
    def __init__(self, structure, learning_rate, activation_function):
        if activation_function == 'sigmoidal':
            self.activation = self.sig
            self.activation_value = self.sig_count
        elif activation_function == 'tanh':
            self.activation = self.tanh
            self.activation_value = self.tanh_count

        self.structure = structure
        self.layer_number = len(structure)
        self.learning_rate = learning_rate
        self.weight_matrix = np.ndarray((self.layer_number - 1), np.object)
        self.activation_matrix = np.ndarray(self.layer_number, np.object)
        self.z_matrix = np.ndarray(self.layer_number, np.object)
        for layerIdx in range(0, self.layer_number - 1):
            # weight_layer = np.random.randn(structure[layerIdx + 1], structure[layerIdx] + 1)
            weight_layer = np.random.uniform(-0.5, 0.5, (structure[layerIdx + 1], structure[layerIdx] + 1))
            self.weight_matrix[layerIdx] = weight_layer
            self.weight_matrix[layerIdx][:, -1] = -1  # todo: sort used

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def sig_count(self, x):
        return (1 - x) * x

    def tanh(self, x):
        return np.exp(x) - np.exp(-x) / np.exp(x) + np.exp(-x)

    def tanh_count(self, x):
        return 1 - x * x

    def decrease(self, arr):

        arr = np.exp(arr)
        arr = arr / np.sum(arr)
        return arr

    def forward_count(self, input_data):
        self.activation_matrix[0] = input_data
        # result = np.ndarray(self.structure[-1])
        for layerIdx in range(0, self.layer_number - 1):
            activation_and_one = np.ones(self.activation_matrix[layerIdx].shape[0] + 1)
            activation_and_one[:-1] = self.activation_matrix[layerIdx]
            self.z_matrix[layerIdx + 1] = np.dot(self.weight_matrix[layerIdx],
                                                 activation_and_one)
            if layerIdx != self.layer_number - 2:
                self.activation_matrix[layerIdx + 1] = self.activation(self.z_matrix[layerIdx + 1])
            else:
                # self.activation_matrix[layerIdx + 1] = self.softmax(self.z_matrix[layerIdx + 1]) # todo: sorting use
                self.activation_matrix[layerIdx + 1] = self.z_matrix[layerIdx + 1]  # todo: fitting use
        result = self.activation_matrix[self.layer_number - 1]
        return result

    def back_count(self, input_data, expectation):
        prediction = self.forward_count(input_data)
        delta_matrix = np.ndarray((self.layer_number - 1), np.object)
        delta_weight = np.ndarray((self.layer_number - 1), np.object)
        for layerIdx in range(0, self.layer_number - 1):
            delta_weight[layerIdx] = np.zeros(self.weight_matrix[layerIdx].shape)
        delta_matrix[-1] = -(expectation - prediction)
        # * self.activation_value(self.activation_matrix[-1])
        for layerIdx in range(self.layer_number - 3, -1, -1):
            z_deri = np.zeros(self.structure[layerIdx + 1])
            for nodeIdx in range(0, self.structure[layerIdx + 1]):
                z_deri[nodeIdx] = self.activation_value(self.activation_matrix[layerIdx + 1][nodeIdx])

            weightMatr = self.weight_matrix[layerIdx + 1].copy()
            weightMatr = np.delete(weightMatr, -1, axis=1)
            weightMatr = np.transpose(weightMatr)
            delta_matrix[layerIdx] = np.dot(weightMatr, delta_matrix[layerIdx + 1]) * z_deri

        for layerIdx in range(0, self.layer_number - 1):
            weight_deri = np.zeros(self.weight_matrix[layerIdx].shape)
            # weight_deri[:, self.structure[layerIdx]] = np.transpose(delta_matrix[layerIdx])
            weight_deri[:, self.structure[layerIdx]] = delta_matrix[layerIdx]
            am = delta_matrix[layerIdx].reshape(-1, 1)
            bm = self.activation_matrix[layerIdx]
            bm = bm.reshape(1, self.structure[layerIdx])
            weight_deri[:, :-1] = np.dot(am, bm)
            delta_weight[layerIdx] += weight_deri
        for layerIdx in range(0, self.layer_number - 1):
            self.weight_matrix[layerIdx] = self.weight_matrix[layerIdx] - self.learning_rate * delta_weight[layerIdx]

    def train_start(self, input_data, output_data, iterations):
        for i in range(iterations):
            self.back_count(input_data, output_data)
