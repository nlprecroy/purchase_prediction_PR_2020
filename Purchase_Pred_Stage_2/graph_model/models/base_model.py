# -*- coding: utf-8 -*-

"""Base Model
"""

import keras
from keras import layers as L
from keras import backend as K
from keras import Model, Sequential

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        # layers
        self.mask_layer = L.Lambda(lambda x: K.cast(
            K.greater(K.expand_dims(x, 2), 0), "float32"))
        self.embedding_layer = L.Embedding(self.config["vocab_size"],
                                           self.config["embedding_size"],
                                           weights=[self.config["embedding_matrix"]],
                                           name="embedding",
                                           #input_length=self.config["max_len"],
                                           trainable=self.config["embeding_trainable"])

        self.dropout_layer = L.Dropout(self.config["dropout"])
        self.reduce_layer = L.Lambda(self.reduce_timestep_maxpooling)
        self.identiy_layer = L.Lambda(self.identity)
        self.output_layer = Sequential([
            L.Dropout(self.config["dropout"]),
            L.Dense(self.config["output_size"], activation="softmax"),
        ])

    def fit(self,
            data_generator,
            steps_per_epoch,
            epochs,
            callbacks=None,
            verbose=1):
        """Fit model"""
        self.model.fit_generator(data_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 verbose=verbose)

    def reduce_timestep_maxpooling(self, x):
        """
        Return:
          - tensor [None, s_size]
        """
        return K.max(x, axis=1, keepdims=False)
    
    def identity(self, x):
        return x

    def eveluate(self, data_generator, steps):
        return self.model.evaluate_generator(data_generator, steps=steps)
    
    def predict(self, data_generator, steps):
        return self.model.predict_generator(data_generator, steps=steps)

    @staticmethod
    def default_params():
        return {
            "embedding_size": 100,
            "dropout": 0.0,
            "optimizer": "adam"
        }
