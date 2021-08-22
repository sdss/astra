import numpy as np

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

class Network:

    def read_in(self, npz_path):
        '''
        read in the weights and biases parameterizing a particular neural network.
        You can read in existing networks from the neural_nets/ directory, or you
        can train your own networks and edit this function to read them in.
        '''

        tmp = np.load(npz_path, allow_pickle=True)
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        self.x_min = tmp["x_min"]
        self.x_max = tmp["x_max"]
        self.NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2)
        self.wave = tmp["wave"]
        if 'grid' in tmp:
            self.grid = tmp["grid"].item()
        tmp.close()

    def num_labels(self):
        return self.NN_coeffs[0].shape[-1]

    def get_spectrum(self, labels):
        scaled = (labels - self.x_min)/(self.x_max - self.x_min) - 0.5
        return self.get_spectrum_scaled(scaled)

    def get_spectrum_scaled(self, scaled_labels):
        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        # assuming your NN has two hidden layers.
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2 = self.NN_coeffs
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
        return spectrum
