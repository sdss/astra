from astra.utils import expand_path
from functools import cache

from astroNN.models import load_folder

@cache
def read_model(model_path):
    return load_folder(expand_path(model_path))