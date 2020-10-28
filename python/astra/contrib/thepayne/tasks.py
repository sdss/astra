
import astra
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, LocalTargetTask)
from astra.tools.spectrum import Spectrum1D
from astra.contrib.thepayne import training, test as testing
from luigi import LocalTarget

class ThePayneMixin(BaseTask):

    task_namespace = "ThePayne"

    n_steps = astra.IntParameter(default=100000)
    n_neurons = astra.IntParameter(default=300)
    weight_decay = astra.FloatParameter(default=0)
    learning_rate = astra.FloatParameter(default=0.001)
    training_set_path = astra.Parameter()


class TrainThePayne(ThePayneMixin):

    """
    Train a single-layer neural network given a pre-computed grid of synthetic spectra.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels
    
    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).
    
    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """
    
    def requires(self):
        """ The requirements of this task."""
        return LocalTargetTask(path=self.training_set_path)


    def run(self):
        """ Execute this task. """

        wavelength, label_names, \
            training_labels, training_spectra, \
            validation_labels, validation_spectra = training.load_training_data(self.input().path)
    
        state, model, optimizer = training.train(
            training_spectra, 
            training_labels,
            validation_spectra,
            validation_labels,
            label_names,
            n_neurons=self.n_neurons,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        with open(self.output().path, "wb") as fp:
            pickle.dump(dict(
                    state=state,
                    wavelength=wavelength,
                    label_names=label_names, 
                ),
                fp
            )

    def output(self):
        """ The output of this task. """
        return LocalTarget(f"{self.task_id}.pkl")
        


class EstimateStellarParameters(ThePayneMixin):
    
    """
    Use a pre-trained neural network to estimate stellar parameters. This should be sub-classed to inherit properties from the type of spectra to be analysed.
    
    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels
    
    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).
    
    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """

    def run(self):
        """ Execute this task. """

        # Load the model.
        state = testing.load_state(self.input()["model"].path)

        # We can run this in batch mode.
        for task in self.get_batch_tasks():

            spectrum = Spectrum1D.read(task.input()["observation"].path)

            p_opt, p_cov, meta = result = testing.test(spectrum, **state)

            with open(self.output().path, "wb") as fp:
                pickle.dump(result, fp)
        
        return None

    def output(self):
        """ The output of this task. """
        return LocalTarget(f"{self.task_id}.pkl")



class EstimateStellarParametersGivenApStarFile(EstimateStellarParameters, ApStarFile):
    """
    Estimate stellar parameters given a single-layer neural network and an ApStar file.

    This task also requiresd all parameters that `astra.tasks.io.ApStarFile` requires.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels
    
    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).
    
    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """

    def requires(self):
        """ The requirements of this task. """
        requirements = {
            "model": TrainThePayne(**self.get_common_param_kwargs(TrainThePayne))
        }
        if not self.is_batch_mode:
            requirements.update(observation=ApStarFile(**self.get_common_param_kwargs(ApStarFile)))
        return requirements

