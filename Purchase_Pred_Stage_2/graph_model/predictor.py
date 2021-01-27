# -*- coding: utf-8 -*-

""" Predictor
"""

import numpy as np
from tqdm import tqdm


class Predictor(object):
    def __init__(self, classifier, dataloader, verbose=0):
        self.classifier = classifier
        self.dataloader = dataloader
        self.verbose = verbose

    def __call__(self, test_data):
        scores, labels = [], []
        if self.verbose == 1:
            test_data = tqdm(test_data)
        for d in test_data:
            x = self.dataloader.token_vector([d["user"]])
            s = self.dataloader.token_vector(d["items"])
            ds = d["disc"]
            proba = self.classifier.model.predict([np.array(x), np.array(s), np.array(ds)])

        return proba