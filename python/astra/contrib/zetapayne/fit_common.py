import matplotlib.pyplot as plt

def save_figure(save_to):
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    fig.savefig(save_to, dpi=200)
    fig.clf()
