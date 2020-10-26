import luigi
import os

from astra.tasks.io import BaseTask, LocalTargetTask
from astra.contrib.classifier import networks, model, utils
from astra.contrib.classifier.tasks.mixin import ClassifierMixin

class TrainSpectrumClassifier(ClassifierMixin, BaseTask):

    """
    A task to train a network to classify source types.

    :param training_spectra_path:
        A path that contains the spectra for the training set.
    
    :param training_set_labels:
        A path that contains the labels for the training set.

    :param validation_spectra_path:
        A path that contains the spectra for the validation set.

    :param validation_labels_path:
        A path that contains the labels for the validation set.

    :param test_spectra_path:
        A path that contains the spectra for the test set.
    
    :param test_labels_path:
        A path that contains ths labels for the test set.

    :param class_names:
        A tuple of names for the object classes.
    
    :param n_epochs: (optional)
        The number of epochs to use for training (default: 200).
    
    :param batch_size: (optional)
        The number of objects to use per batch in training (default: 100).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 1e-5).
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 1e-4).
    """

    def requires(self):
        """ The dependencies required by this task. """
        return {
            "training_spectra": LocalTargetTask(path=self.training_spectra_path),
            "training_labels": LocalTargetTask(path=self.training_labels_path),
            "validation_spectra": LocalTargetTask(path=self.validation_spectra_path),
            "validation_labels": LocalTargetTask(path=self.validation_labels_path),
            "test_spectra": LocalTargetTask(path=self.test_spectra_path),
            "test_labels": LocalTargetTask(path=self.test_labels_path)
        }


    def output(self):
        """ The outputs generated by this task. """
        # By default place it relative to the input path of the training spectra.
        output_path_prefix, ext = os.path.splitext(self.input()["training_spectra"].path)
        return luigi.LocalTarget(f"{output_path_prefix}-{self.task_id}.model")


    def run(self):
        """
        Run the task.
        """

        try:
            network = self.network_factory

        except AttributeError:
            raise RuntimeError("You should only be using TrainOpticalClassifier or TrainNIRClassifier")

        # Load all them data.
        training_spectra, training_labels = utils.load_data(
            self.input()["training_spectra"].path,
            self.input()["training_labels"].path
        )

        validation_spectra, validation_labels = utils.load_data(
            self.input()["validation_spectra"].path,
            self.input()["validation_labels"].path
        )

        test_spectra, test_labels = utils.load_data(
            self.input()["test_spectra"].path,
            self.input()["test_labels"].path
        )

        state, network, optimizer = model.train(
            self.network_factory,
            training_spectra,
            training_labels,
            validation_spectra,
            validation_labels,
            test_spectra,
            test_labels,
            class_names=self.class_names,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            task=self
        )

        # Write the model to disk.
        utils.write_network(
            network,
            self.output().path
        )


class TrainNIRSpectrumClassifier(TrainSpectrumClassifier):
    network_factory = networks.NIRCNN


class TrainOpticalSpectrumClassifier(TrainSpectrumClassifier):
    network_factory = networks.OpticalCNN