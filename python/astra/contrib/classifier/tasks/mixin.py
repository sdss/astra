import astra
from astra.tasks import BaseTask
from astra.tasks.slurm import slurm_mixin_factory

SlurmMixin = slurm_mixin_factory("Classifier")

class ClassifierMixin(SlurmMixin, BaseTask):

    """ 
    A mix-in class for classifier parameters. 
    
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

    task_namespace = "Classifier"

    training_spectra_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="training_spectra_path")
    )
    training_labels_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="training_labels_path")
    )

    validation_spectra_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="validation_spectra_path")
    )
    validation_labels_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="validation_labels_path")
    )
    
    test_spectra_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="test_spectra_path")
    )
    test_labels_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="test_labels_path")
    )

    #class_names = astra.ListParameter(
    #    config_path=dict(section=task_namespace, name="class_names"),
    #    significant=False
    #)

    n_epochs = astra.IntParameter(default=200)
    batch_size = astra.IntParameter(default=100)
    weight_decay = astra.FloatParameter(default=1e-5)
    learning_rate = astra.FloatParameter(default=1e-4)

    max_batch_size = 10_000

    def get_tqdm_kwds(self, desc=None):
        kwds = dict(
            iterable=self.get_batch_tasks(),
            total=self.get_batch_size(),
            desc=desc
        )
        if kwds["total"] == 1: kwds.update(disable=True)
        return kwds

    @property
    def class_names(self):
        """ Names of classes used in this classifier. """
        # TODO: requires a thinko.
        return ('fgkm', 'hotstar', 'sb2', 'yso')
