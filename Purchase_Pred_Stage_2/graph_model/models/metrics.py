# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from graph_model.predictor import Predictor
from decimal import *
from keras.models import Model

def f1_metric(y_true, y_pred):
    """ calc f1 metric
    code from: https://stackoverflow.com/a/45305384/5210098
    """
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



class Metrics(Callback):
    def __init__(self, train_data, validation_data, train_step, validtion_step):
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.train_step = train_step
        self.validtion_step = validtion_step
        self.history = defaultdict(list)

    def calc_metrics(self, generator, step, is_test):
        def _sigmoid(x):
            return 1/(1+np.exp(-x))
        y_true, y_pred = [], []
        cnt = 0
        for X, y in generator:
            if cnt > step:
                break
            cnt += 1
            #y = np.argmax(y, axis=1)
            pred = self.model.predict(X)

            #pred = np.argmax(pred, 1)
            pred_ = []

            for each in pred:
                for i in _sigmoid(each):
                    if i > 0.4:   ################ This is a hyperparameter and can be adjusted accordingly. In this work, the model produces the best results with i > 0.4.
                        pred_.append(1)
                    else:
                        pred_.append(0)
            true_ = []
            for each in y:
                for i in each:
                    if i == 1:
                        true_.append(1)
                    else:
                        true_.append(0)
            y_true += list(true_)
            y_pred += list(pred_)
        print(len(y_pred))
        print(sum(y_pred))
        print(sum(y_true))
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print("accuracy", acc)
        print("f1", f1)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true_1 = y_true[np.where(y_true == 1)]
        y_pred_1 = y_pred[np.where(y_true == 1)]

        acc_1 = accuracy_score(y_true_1, y_pred_1)
        f1_1 = float(classification_report(y_true, y_pred, digits=4).split("\n")[3].split(" ")[24])
        
        if is_test:
            print(classification_report(y_true, y_pred, digits=4))
            print(confusion_matrix(y_true, y_pred))
        return f1, acc, f1_1, acc_1

    def on_epoch_end(self, epoch, logs={}):
        val_f1, val_acc, val_f1_1, val_acc_1 = self.calc_metrics(self.validation_data, self.validtion_step, is_test = True)
        train_f1, train_acc, _, _ = self.calc_metrics(self.train_data, self.train_step, is_test = False)
        print(f"- train_acc {round(train_acc, 5)} - train_f1 {round(train_f1, 5)} - val_acc {round(val_acc, 5)} - val_f1 {round(val_f1, 5)}")
    
        #print(self.model.layers[2].get_weights()[0].shape)


        self.history["train_acc"].append(train_acc)
        self.history["train_f1"].append(train_f1)
        self.history["val_acc"].append(val_acc)
        self.history["val_f1"].append(val_f1)
        self.history["val_acc_1"].append(val_acc_1)
        self.history["val_f1_1"].append(val_f1_1)
        #self.history["embedding"].append(self.model.layers[2].get_weights()[0])


