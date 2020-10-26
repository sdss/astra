import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues
    ):
    """
    Plot a confusion matrix.

    :param y_true:
        An array of true (or believed) classes.
    
    :param y_pred:
        An array of predicted classes.
    
    :param class_names:
        The names associated with each class integer.

    :param normalize: (optional)
        Normalize the confusion matrix to show fractions instead of absolute values (default: False).
    
    :param title: (optional)
        A title for the figure.
    
    :param cmap: (optional)
        A colormap for the confusion matrix (default: `plt.cm.Blues`).
    
    :returns:
        A figure.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    y_true_ = [int(y) for y in y_true]
    y_pred_ = [int(y) for y in y_pred]
    
    cm = confusion_matrix(y_true_, y_pred_)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, 
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), 
        rotation=45, 
        ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, 
                i, 
                format(cm[i, j], fmt),
                ha="center", 
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    ax.set_ylim(cm.shape[0] - 0.5, - 0.5)
    fig.tight_layout()
    return fig