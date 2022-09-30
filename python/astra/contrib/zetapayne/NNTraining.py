'''
This file is used to train the neural net that predicts the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is slightly different from Ting+19. Instead of
training individual small networks for each pixel separately, here we train a single
large network for all pixels simultaneously.

The advantage of doing so is that individual pixels will exploit information
from adjacent pixels. This usually leads to more precise interpolations.

However to train a large network, GPU is needed. This code will
only run with GPU. But even with an inexpensive GPU, this code
should be pretty efficient -- training with a grid of 10,000 training spectra,
with > 10 labels, should not take more than a few hours

The default training set are the Kurucz synthetic spectral models and have been
convolved to the appropriate R (~22500 for APOGEE) with the APOGEE LSF.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
import time
from torch.autograd import Variable
#from RAdam import RAdam
#from bisect import bisect
from astra.contrib.zetapayne.RAdam import RAdam
from astra.contrib.zetapayne.bisect import bisect

#===================================================================================================
# simple multi-layer perceptron model
class Perceptron(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_pixel):
        super(Perceptron, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)

class PerceptronSP(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_pixel):
        super(PerceptronSP, self).__init__()

        self.features = torch.nn.ModuleList()
        for i in range(num_pixel):
            s = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, 1),)
            self.features.append(s)

    def forward(self, x):
        bs = x.shape[0]
        y = torch.zeros(bs, len(self.features))
        for i,v in enumerate(self.features):
            q = v(x)
            for j in range(bs):
                y[j,i] = q[j]
        return y



#---------------------------------------------------------------------------------------------------

class NNTraining:
    '''
    Training a neural net to emulate spectral models

    training_labels has the dimension of [# training spectra, # stellar labels]
    training_spectra has the dimension of [# training spectra, # wavelength pixels]

    The validation set is used to independently evaluate how well the neural net
    is emulating the spectra. If the neural network overfits the spectral variation, while
    the loss will continue to improve for the training set, but the validation
    set should show a worsen loss.

    The training is designed in a way that it always returns the best neural net
    before the network starts to overfit (gauged by the validation set).

    num_steps = how many steps to train until convergence.
    1e4 is good for the specific NN architecture and learning I used by default.
    Bigger networks will take more steps to converge, and decreasing the learning rate
    will also change this. You can get a sense of how many steps are needed for a new
    NN architecture by plotting the loss evaluated on both the training set and
    a validation set as a function of step number. It should plateau once the NN
    has converged.

    learning_rate = step size to take for gradient descent
    This is also tunable, but 1e-4 seems to work well for most use cases. Again,
    diagnose with a validation set if you change this.

    num_features is the number of features before the deconvolutional layers; it only
    applies if ResNet is used. For the simple multi-layer perceptron model, this parameter
    is not used. We truncate the predicted model if the output number of pixels is
    larger than what is needed. In the current default model, the output is ~8500 pixels
    in the case where the number of pixels is > 8500, increase the number of features, and
    tweak the ResNet model accordingly

    batch_size = the batch size for training the neural networks during the stochastic
    gradient descent. A larger batch_size reduces stochasticity, but it might also
    risk of stucking in local minima

    '''

    def __init__(self, num_neurons = 300, num_steps=10000, learning_rate=1.e-4, batch_size=64,\
             num_features = 64*5, mask_size=11, batch_size_valid=64):
        self.num_neurons = num_neurons
        self.num_steps = int(num_steps)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_features = num_features
        self.mask_size = mask_size
        self.batch_size_valid = batch_size_valid 
        self.CUDA = torch.cuda.is_available()

    def train_on_npz(self, npz_path, wave_range, validation_fraction=0.1):
        self.vf = validation_fraction
        data = np.load(npz_path, allow_pickle=True)
        wave = data['wvl']
        i_start = bisect(wave, wave_range[0])
        i_end = bisect(wave, wave_range[1])
        print('Wave index range:', i_start, i_end)
        spectra = np.squeeze(data['flux'])[:,i_start:i_end]
        labels  = data['labels']
        self.wave = wave[i_start:i_end]
        print('spectra', spectra.shape)
        print('labels', labels.shape)
        N_total = spectra.shape[0]
        N_valid = int(N_total * validation_fraction)
        print('total/valid:', N_total, '/', N_valid)

        idx = torch.randperm(N_total)
        idx_valid = idx[:N_valid]
        idx_train = idx[N_valid:]

        validation_spectra = spectra[idx_valid,:]
        validation_labels = labels[idx_valid,:]
        training_spectra = spectra[idx_train,:]
        training_labels = labels[idx_train,:]

        if self.batch_size_valid>N_valid:
            self.batch_size_valid = N_valid

        self.grid_info = data['grid'].item()

        self.train(training_labels, training_spectra, validation_labels, validation_spectra)


    def train(self, training_labels, training_spectra, validation_labels, validation_spectra):
        print('start training')
        # run on cuda
        if self.CUDA:
            self.dtype = torch.cuda.FloatTensor
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.dtype = torch.FloatTensor
            torch.set_default_tensor_type('torch.FloatTensor')
        print('cuda tensor type set')
        x, y, x_valid, y_valid = self.scale_variables(training_labels, training_spectra, validation_labels, validation_spectra)
        print('variables scaled')

        # assume L1 loss
        self.loss_fn = torch.nn.L1Loss(reduction = 'mean')
        # initiate Payne and optimizer
        model = Perceptron(x.shape[1], self.num_neurons, training_spectra.shape[1])
        if self.CUDA:
            model.cuda()
        model.train()
        self.model = model

        # we adopt rectified Adam for the optimization
        self.optimizer = RAdam([p for p in model.parameters() if p.requires_grad==True], lr=self.learning_rate)

    #--------------------------------------------------------------------------------------------
        # train in batches
        self.batch_init(x, x_valid)
        print('batch initialized')

        for e in range(self.num_steps):
            self.step(x, y)
            # evaluate validation loss
            if e % 100 == 0:
                self.validate(x_valid, y_valid)
                print('iter %s:' % e, 'training loss = %.3f' % self.loss,\
                     'validation loss = %.3f' % self.loss_valid)



    def batch_init(self, x, x_valid):
        self.nsamples = x.shape[0]
        self.nbatches = self.nsamples // self.batch_size

        self.nsamples_valid = x_valid.shape[0]
        self.nbatches_valid = self.nsamples_valid // self.batch_size_valid

        # initiate counter
        self.current_loss = np.inf
        self.training_loss =[]
        self.validation_loss = []

    def step(self, x, y):
        # randomly permute the data
        perm = torch.randperm(self.nsamples)
        if self.CUDA:
            perm = perm.cuda()

        # for each batch, calculate the gradient with respect to the loss
        for i in range(self.nbatches):
            idx = perm[i * self.batch_size : (i+1) * self.batch_size]
            y_pred = self.model(x[idx])

            self.loss = self.loss_fn(y_pred, y[idx])*1.e4
            self.optimizer.zero_grad()
            self.loss.backward(retain_graph=False)
            self.optimizer.step()


    def validate(self, x_valid, y_valid):

        # here we also break into batches because when training ResNet
        # evaluating the whole validation set could go beyond the GPU memory
        # if needed, this part can be simplified to reduce overhead
        perm_valid = torch.randperm(self.nsamples_valid)
        if self.CUDA:
            perm_valid = perm_valid.cuda()

        loss_valid = 0
        for j in range(self.nbatches_valid):
            idx = perm_valid[j * self.batch_size_valid : (j+1) * self.batch_size_valid]
            y_pred_valid = self.model(x_valid[idx])
            loss_valid += self.loss_fn(y_pred_valid, y_valid[idx])*1.e4
        loss_valid /= self.nbatches_valid
        self.loss_valid = loss_valid

        loss_data = self.loss.detach().data.item()
        loss_valid_data = loss_valid.detach().data.item()
        self.training_loss.append(loss_data)
        self.validation_loss.append(loss_valid_data)

#--------------------------------------------------------------------------------------------
        # record the weights and biases if the validation loss improves
        if loss_valid_data < self.current_loss:
            self.current_loss = loss_valid_data
            self.save_NN()
            # save the training loss
            np.savez("training_loss.npz",\
                     training_loss = self.training_loss,\
                     validation_loss = self.validation_loss)


    def scale_variables(self, training_labels, training_spectra, validation_labels, validation_spectra):
        dtype = self.dtype
        # scale the labels, optimizing neural networks is easier if the labels are more normalized
        x_max = np.max(training_labels, axis = 0)
        x_min = np.min(training_labels, axis = 0)
        self.x_max = x_max
        self.x_min = x_min
        x = (training_labels - x_min)/(x_max - x_min) - 0.5
        x_valid = (validation_labels-x_min)/(x_max-x_min) - 0.5

        # make pytorch variables
        print('creating pytorch variables')
        x = Variable(torch.from_numpy(x)).type(dtype)
        y = Variable(torch.from_numpy(training_spectra), requires_grad=False).type(dtype)
        x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
        y_valid = Variable(torch.from_numpy(validation_spectra), requires_grad=False).type(dtype)

        return x, y, x_valid, y_valid

    def save_NN(self):
        model_numpy = []
        for param in self.model.parameters():
            model_numpy.append(param.data.cpu().numpy())

        # extract the weights and biases
        w_array_0 = model_numpy[0]
        b_array_0 = model_numpy[1]
        w_array_1 = model_numpy[2]
        b_array_1 = model_numpy[3]
        w_array_2 = model_numpy[4]
        b_array_2 = model_numpy[5]

        # save parameters and remember how we scaled the labels
        fn = 'NN_n%i_b%i_v%.1f.npz'%(self.num_neurons, self.batch_size, self.vf)
        np.savez(fn,\
                w_array_0 = w_array_0,\
                w_array_1 = w_array_1,\
                w_array_2 = w_array_2,\
                b_array_0 = b_array_0,\
                b_array_1 = b_array_1,\
                b_array_2 = b_array_2,\
                x_max=self.x_max,\
                x_min=self.x_min,
                wave = self.wave,
                grid=self.grid_info,)














