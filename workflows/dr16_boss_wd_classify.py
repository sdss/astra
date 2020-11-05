import astra
import numpy as np
import os
import yaml
from astra.contrib.wd.tasks.classify import ClassifyWhiteDwarfGivenSpecFile

directory = "../astra-components/data/wd/"

# Get the expected classes for 
with open(os.path.join(directory, "sdss-wd-examples.yml"), "r") as fp:
    examples = yaml.load(fp, Loader=yaml.FullLoader)

# Let's separate the expected classes and join the keywords together.
N = None # Optionally only do a subset of the data.
kwds = { key: [] for key in examples[0].keys() }
for i, each in enumerate(examples, start=1):
    if N is not None and i >= N: break
    for k, v in each.items():
        kwds[k].append(v)

expected_classes = kwds.pop("expected_class")
kwds.update(
    release="dr16",
    model_path=os.path.join(directory, "training_file")
)

# Create the task where we will run everything in batch mode.
task = ClassifyWhiteDwarfGivenSpecFile(**kwds)

# Build the acyclic graph.
astra.build(
    [task],
    local_scheduler=True
)

import pickle

predicted_classes = []
for output in task.output():
    with open(output.path, "rb") as fp:
        result = pickle.load(fp)
    predicted_classes.append(result["wd_class"])

# Let's plot a confusion matrix.

# Restrict to major classes only.
max_chars = 2
unique_class_names = sorted(set([pc[:max_chars] for pc in predicted_classes]))

M = len(unique_class_names)
confusion_matrix = np.zeros((M, M))
for i, (predicted_class, expected_class) in enumerate(zip(predicted_classes, expected_classes)):
    j = unique_class_names.index(expected_class.upper())
    k = unique_class_names.index(predicted_class.upper()[:max_chars])
    confusion_matrix[j, k] += 1

