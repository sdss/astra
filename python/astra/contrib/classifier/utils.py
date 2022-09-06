import numpy as np
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_data(spectra_path, labels_path):
    """
    Load data for training, testing, or validation.

    :param spectra_path:
        The path containing the spectra. This should be a pickled array that is readable with `np.load`.

    :param labels_path:
        The path containing the labels. This should be a pickled array that is readable with `np.load`.

    :returns:
        A two-length tuple containing the spectra and labels.
    """

    with open(spectra_path, "rb") as fp:
        spectra = np.load(fp)

    with open(labels_path, "rb") as fp:
        labels = np.load(fp)

    if spectra.shape[0] != labels.size:
        raise ValueError(
            f"spectra and labels have different shapes ({spectra.shape[0]} != {labels.size})"
        )

    return (spectra, labels)


def norm_spectra_med(signal):
    """
    Pseudo-continuum-normalise a signal based on the median.
    """
    return [signal[i, :] / np.median(signal, axis=1)[i] for i in range(signal.shape[0])]


def write_network(network, path):
    """
    Write a neural network state to disk.

    :param network:
        The neural network to save.

    :param path:
        The path to save the neural network to.
    """
    torch.save(network.state_dict(), path)


def read_network(network_class, path):
    """
    Read a neural network state from disk.

    :param network_class:
        The network class factory to load the model.

    :param path:
        The path where the neural network coefficients are stored.

    :returns:
        The loaded neural network.
    """
    network = network_class()
    network.load_state_dict(torch.load(path, map_location=device))
    return network
