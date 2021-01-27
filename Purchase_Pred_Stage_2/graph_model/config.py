# -*- coding: utf-8 -*-

import os


class Config:
    def __init__(self, config):
        self.default_config = {
            "batch_size": 16,  # batch size
            "cv_fold": 10,
            "embeding_trainable": True,
            "gridsearch": False,
            "split_round": 5,  # how many rounds to train [for split data set]
            "trigger_accuray": None,
            "epochs": 10,  # epochs to train.
            "verbose": 1,  # show training logs. 0 not 1 show,
        }

        self.config = dict(self.default_config, **config)

    def __call__(self):
        if "log_path" not in self.config:
            raise ValueError("`log_path` is required")
        if "pretrained_embedding_path" not in self.config:
            raise ValueError("`pretrained_embedding_path` is required")
        return self.config
