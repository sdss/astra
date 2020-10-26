import luigi

class ClassifierMixin(luigi.Config):

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

    training_spectra_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="training_spectra_path")
    )
    training_labels_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="training_labels_path")
    )

    validation_spectra_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="validation_spectra_path")
    )
    validation_labels_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="validation_labels_path")
    )
    
    test_spectra_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="test_spectra_path")
    )
    test_labels_path = luigi.Parameter(
        config_path=dict(section="ClassifierTask", name="test_labels_path")
    )

    class_names = luigi.ListParameter(
        config_path=dict(section="ClassifierTask", name="class_names"),
        significant=False
    )

    n_epochs = luigi.IntParameter(default=200)
    batch_size = luigi.IntParameter(default=100)
    weight_decay = luigi.FloatParameter(default=1e-5)
    learning_rate = luigi.FloatParameter(default=1e-4)
