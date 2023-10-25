import numpy as np
import torch
from scipy.special import logsumexp


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def classification_result(log_probs, class_names=None, decimals=30):

    if class_names is None:
        if not isinstance(log_probs, dict):
            raise TypeError(
                f"If class_names is None then log_probs must be a dictionary"
            )
        class_names, log_probs = zip(*log_probs.items())

    log_probs = np.array(log_probs).flatten()
    # Calculate normalized probabilities.
    with np.errstate(under="ignore"):
        relative_log_probs = log_probs - logsumexp(log_probs)#, axis=1)[:, None]

    # Round for PostgreSQL 'real' type.
    # https://www.postgresql.org/docs/9.1/datatype-numeric.html
    # and
    # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
    probs = np.round(np.exp(relative_log_probs), decimals)
    log_probs = np.round(log_probs, decimals)

    result = {f"p_{cn}": p for cn, p in zip(class_names, probs.T)}
    result.update({f"lp_{cn}": p for cn, p in zip(class_names, log_probs.T)})
    result[f"flag_most_likely_{class_names[np.argmax(probs)]}"] = True
    return result

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
