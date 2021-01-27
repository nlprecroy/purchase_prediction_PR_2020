# -*- coding: utf-8 -*-

""" Deep Classifier
"""

import os
import json
import time
import pickle as pkl
from pprint import pprint

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid

from graph_model.config import Config
from graph_model.dataloader import DataLoader, default_tokenizer
from graph_model.models import *
from graph_model.models.metrics import Metrics

__version__ = "0.1.0"
__author__ = 'zongxi li'
__email__ = 'zongxili2@gmail.com'


def load(save_dir, tokenizer=None):
    """ Load model from pretrained model
    """
    config_path = os.path.join(save_dir, "config.json")
    config = GraphModel.load_config(config_path)
    config["save_dir"] = save_dir
    config = Config(config)()
    model = GraphModel(tokenizer=tokenizer, **config)
    return model


class GraphModel:
    def __init__(self, is_training=False, tokenizer=None, **config):
        self.config = Config(config)()
        self.tokenizer = default_tokenizer if tokenizer is None else tokenizer
        self.dataloader = DataLoader(self.config, tokenizer=self.tokenizer)
        self.data_mode = "cv"  # cv, split
        self.config["mt_weight"] = None
        self.model_instance = None
        if self.config["model"].lower() == "purchase":
            self.model_instance = Purchase
        else:
            raise NotImplementedError(f"""Error! dont support the input model
                    Only support [`purchase`] now.""")

        default_params = self.model_instance.default_params()
        self.config = dict(default_params, **self.config)

        if "data_path" in self.config and self.config["data_path"]:
            self.data_mode = "cv"
            self.dataloader.set_data(self.config["data_path"])
        else:
            if "train_path" not in self.config or "test_path" not in self.config:
                raise ValueError("`train_path` and `test_path` is required")
            self.data_mode = "split"
            self.dataloader.set_data(self.config["train_path"], "train")
            self.dataloader.set_data(self.config["test_path"], "test")

        # load pretrained embedding
        embedding_matrix, embedding_size = self.dataloader.load_pretrained_embedding(self.config["pretrained_embedding_path"])
        self.config["embedding_matrix"] = embedding_matrix
        self.config["embedding_size"] = embedding_size
        self.config["vocab_size"] = self.dataloader.vocab.token_size
        self.config["output_size"] = self.dataloader.vocab.label_size ####Need to change as a parameter but have to be less than 10
        pprint(self.config)
    


    def train(self, verbose=2):
        """ Train model
        Args:
          - verbose: int
        """
        if self.data_mode == "cv":
            self.train_cv(verbose=verbose)
        else:
            print("Only Support Cross-Validation")
            exit()


    def train_cv(self, verbose=2):
        model_name = self.config["model"].lower()
        pretrain_path = self.config["pretrained_embedding_path"].split("/")[-1]
        accuracies = []
        f1s = []
        accuracies_1 = []
        f1s_1 = []
        histories = []
        histories_1 = []
        start_time = time.time()
        with open(self.config["log_path"], "w") as writer:
            for idx in range(self.config["cv_fold"]):
                print(f"start to fit fold {idx+1}...")
                best_round = [0., 0.]
                best_round_1 = [0., 0.]
                best_history = None
                best_history_1 = None
                history_embeddings = []
                self.dataloader.set_train_test(idx)
                train_generator = self.dataloader.generator("train", self.config["batch_size"])
                dev_generator = self.dataloader.generator("dev", self.config["batch_size"], is_shuffle=False)
                test_generator = self.dataloader.generator("test", self.config["batch_size"], is_shuffle=False)
                
                
                #with open("saved_embedding/vocab.pkl", "wb") as writer_voc:
                    #pkl.dump(self.dataloader.vocab.token2idx, writer_voc)
                #print("save vocabulary")  
                iters = self.config["iterperfold"]
                for i in range(iters):
                    initial_log = f"fold {idx+1} round {i+1}/{iters}"
                    print(initial_log)
                    classifier = self.model_instance(self.config)
                    '''
                    history = classifier.model.fit_generator(train_generator,
                                                             steps_per_epoch=self.dataloader.steps_per_epoch,
                                                             epochs=self.config["epochs"],
                                                             validation_data=test_generator,
                                                             validation_steps=self.dataloader.test_steps,
                                                             verbose=verbose)
                    accuracy = max(history.history["val_acc"])
                    f1 = max(history.history["val_f1_metric"])
                    '''
                    metrics_callback = Metrics(train_data=train_generator,
                                               validation_data=test_generator,
                                               train_step=self.dataloader.steps_per_epoch,
                                               validtion_step=self.dataloader.test_steps)
                    history = classifier.model.fit_generator(train_generator,
                                                             steps_per_epoch=self.dataloader.steps_per_epoch,
                                                             epochs=self.config["epochs"],
                                                             callbacks=[metrics_callback],
                                                             verbose=verbose)
                    
                    history.history = dict(history.history, **metrics_callback.history)
                    
                    embedding_weight = metrics_callback.history["embedding"]
                    #history_embeddings.append(embedding_weight)    

                    accuracy = max(metrics_callback.history["val_acc"])
                    f1 = max(metrics_callback.history["val_f1"])
                    print(metrics_callback.history["val_acc_1"])
                    accuracy_1 = max(metrics_callback.history["val_acc_1"])
                    f1_1 = max(metrics_callback.history["val_f1_1"])                   
                    if accuracy + f1 > sum(best_round):
                        #best_round = [accuracy, f1]
                        best_history = history.history
                    if accuracy > best_round[0]:
                        best_round[0] = accuracy
                    if f1 > best_round[1]:
                        best_round[1] = f1

                    if accuracy_1 + f1_1 > sum(best_round_1):
                        #best_round = [accuracy, f1]
                        best_history_1  = history.history
                    if accuracy_1  > best_round_1[0]:
                        best_round_1[0] = accuracy_1
                    if f1_1 > best_round_1[1]:
                        best_round_1[1] = f1_1

                    log = f"fold {idx+1} round {i+1}/{iters} accuracy: {accuracy}, f1: {f1},  accuracy_1: {accuracy_1}, f1_1: {f1_1}\n"
                    writer.writelines(log)
                    print(log)
                
                #with open("saved_embedding/all_embeddings.pkl", "wb") as writer_emb:
                    #pkl.dump(embedding_weight, writer_emb)
                #print("save embedding")  
                
                histories.append(best_history)
                accuracies.append(best_round[0])
                f1s.append(best_round[1])

                histories_1.append(best_history_1)
                accuracies_1.append(best_round_1[0])
                f1s_1.append(best_round_1[1])

                log = f"fold {idx+1} accuracy: {best_round[0]}, f1: {best_round[1]},  accuracy_1: {best_round_1[0]}, f1_1: {best_round_1[1]}\n"
                writer.writelines(log)
                print("-----------------------------")
                print(log)
                print("=============================")
                print(accuracies)
                current_avg_accu = sum(accuracies_1) / len(accuracies_1)
                print("current avg accu is")
                print(current_avg_accu)
                print("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
            avg_accu = sum(accuracies) / len(accuracies)
            std_accu = np.std(accuracies)
            avg_f1 = sum(f1s) / len(f1s)
            std_f1 = np.std(f1s)

            avg_accu_1 = sum(accuracies_1) / len(accuracies_1)
            avg_f1_1   = sum(f1s_1) / len(f1s_1)

            cost_time = int(time.time() - start_time)
            log = f"average accuracy: {avg_accu}, average f1: {avg_f1}, average accuracy of 1: {avg_accu_1}, average f1 of 1: {avg_f1_1}, time: {cost_time} s\n"
            writer.writelines(log)
            print(log)
            print(self.config["model"])
            print("------")
            print(accuracies)
            '''
            with open("history.pkl", "wb") as his_writer:
                print("save history...")
                obj = {
                    "history": histories,
                    "eval_accuracy": accuracies,
                    "f1": f1s
                }
                pkl.dump(obj, his_writer)
            '''
            with open("results_history.txt", "a") as his_writer:
                print("Save history ...")
                log = f"{model_name} average accuracy: {avg_accu} (std: {std_accu}), average f1: {avg_f1} (std: {std_f1}), average accuracy of 1: {avg_accu_1}, average f1 of 1: {avg_f1_1},time: {cost_time} s, accu: {accuracies}, f1: {f1s}, accu_1: {accuracies_1}, f1_1: {f1s_1}, {pretrain_path} \n"
                his_writer.writelines(log)


    def calculate_f1(self, model, validation_data, validation_steps):
        y_true, y_pred = [], []
        step = 0
        for (x, y) in validation_data:
            if step > validation_steps:
                break
            step += 1
            y_true += y
            pred = model.predict(x)
            pred = np.argmax(pred)
            y_pred += pred
            print("y true", y_true)
            print("y pred", y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        return f1


    @staticmethod
    def load_config(config_path):
        """ Load config
        Args:
          - config_path: path to config
        """
        with open(config_path, "r", encoding="utf-8") as reader:
            return json.load(reader)

    def save_config(self):
        """ Save config to file
        """
        with open(self.config["config_path"], "w", encoding="utf-8") as writer:
            json.dump(self.config, writer, ensure_ascii=False, indent=2)
