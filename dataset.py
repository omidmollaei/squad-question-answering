"""load and preprocess SQuAD dataset, build tokenizer and convert dataset into a tf-dataset"""

import os
import datasets
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class Parameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)