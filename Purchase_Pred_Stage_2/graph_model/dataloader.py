# -*- coding: utf-8 -*-

"""DataLoader
"""

import re
import json
import random
import math
from pprint import pprint
from collections import defaultdict
import pickle as pl
import numpy as np
from boltons.iterutils import chunked
from keras.utils import to_categorical
from nltk.corpus import stopwords, brown

def default_tokenizer(text):
    """ default tokenizer
    """
    return text.split()


class Vocabulary:
    def __init__(self):
        self._token2label = defaultdict(dict)
        self.token2idx = {}
        self.idx2token = {}
        self.label_size = 10

    def get_token_idx(self, token):
        """ get token index
        """
        return self.token2idx.get(token, self.token2idx['<unk>'])

    @property
    def token_size(self):
        return len(self.token2idx) + 1



class DataLoader:
    def __init__(self, config, fold=10, tokenizer=None):
        """dataloader init
        Args:
          - tokenizer:
        """
        self.config = config
        self.dataset = None
        self.cv_datasets = None
        self.fold = fold
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.train_steps = None
        self.dev_steps = None
        self.test_steps = None
        self.vocab = Vocabulary()

        if tokenizer is None:
            self.tokenizer = default_tokenizer
        else:
            self.tokenizer = tokenizer

    def set_data(self, filepath, setname=None):
        """set train dataset
        Args:
          - filepath: dataset path
        """
        dataset = self.read_data(filepath, is_build_vocab=False)
        if setname == "train":
            self.train_set = dataset
            self.train_steps = len(self.train_set) // self.config["batch_size"]
        elif setname == "test":
            self.test_set = dataset
            self.test_steps = len(self.test_set) // self.config["batch_size"]
        elif setname == "dev":
            self.dev_set = dataset
            self.dev_steps = len(self.dev_set) // self.config["batch_size"]
        else:
            self.dataset = dataset
            self.cv_datasets = chunked(dataset, math.ceil(len(dataset) / self.fold))

    def set_dev_set(self):
        if self.dev_set is None and self.train_set is not None:
            # split new train and dev set from old train set
            train_dev_set = self.train_set
            self.dev_set = [v for i, v in enumerate(train_dev_set) if i % 10 == 0]
            self.train_set = [v for i, v in enumerate(train_dev_set) if i % 10 != 0]
            self.train_steps = len(self.train_set) // self.config["batch_size"]
            self.dev_steps = len(self.dev_set) // self.config["batch_size"]

    def set_train_test(self, test_idx, is_gridsearch=False):
        self.test_set = self.cv_datasets[test_idx]
        train_dev_set = []
        for chunk in self.cv_datasets[:test_idx] + self.cv_datasets[test_idx+1:]:
            train_dev_set.extend(chunk)
        if is_gridsearch:
            random.shuffle(train_dev_set)
            self.dev_set = [v for i, v in enumerate(train_dev_set) if i % 10 == 0]
            self.train_set = [v for i, v in enumerate(train_dev_set) if i % 10 != 0]
            self.dev_steps = len(self.dev_set) // self.config["batch_size"]
            print("dev size>>>>",len(self.dev_set))
        else:
            self.train_set = train_dev_set
        print("train size>>>>",len(self.train_set))
        print("test size>>>>",len(self.test_set))
        self.train_steps = len(self.train_set) // self.config["batch_size"]
        self.test_steps = len(self.test_set) // self.config["batch_size"]

    def read_data(self, filepath, is_build_vocab=False):
        """read dataset from file
        Args:
          - filepath: dataset path
          - is_build_vocab: whether to build vocab
        """

        with open("general_list.pkl", "rb") as file:
            self.general_list = pl.load(file)
        self.vocab.token2idx = {"<pad>": 0, "<unk>": 1}
        print(len(self.general_list))
        ll = 2
        for token in self.general_list:
            self.vocab.token2idx[token] = ll
            ll+=1

        print("max id", max(list(self.vocab.token2idx.values())), len(self.vocab.token2idx))
        self.vocab.idx2token = {idx: token for token, idx in self.vocab.token2idx.items()}
        #print("max_len", self.vocab.token2idx)
        datas = []

        with open(filepath, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                datas.append(obj)

        return datas

    def token_vector(self, tokens):
        """ get token vector
        """
        return [self.vocab.get_token_idx(t) for t in tokens]



    def generator(self, setname, batch_size, is_shuffle=True):
        """ build data generator
        """
        dataset = None
        if setname == "train":
            dataset = self.train_set
        elif setname == "test":
            dataset = self.test_set
        elif setname == "dev":
            dataset = self.dev_set
        else:
            raise NotImplementedError(f"""Error! dont support setname `{setname}`.
                    Only support [`train`, `test`] now.""")
        while True:
            if is_shuffle:
                idxs = np.random.permutation(len(dataset))
            else:
                idxs = range(len(dataset))
            u, s, ds, y = [], [], [], []
            for idx in idxs:
                d = dataset[idx]

                u.append(self.token_vector([d["user"]])*10)

                s.append(self.token_vector(d["items"]))
                labs = []
                #print([int(lab) for lab in d["label"]])
                for each in [int(lab) for lab in d["label"]]:
                    if each == 1:
                        labs.append(1)
                    else:
                        labs.append(-1)
                y.append(labs)
                _discount = [float(n) for n in d["disc"]]
                discount = []
                for i in range(10):
                    curr_dis =_discount[6*i: 6*i+6]
                    curr_dis_ = []
                    if (curr_dis[0] != 0) & (curr_dis[0]!=curr_dis[1]):
                        #print(curr_dis)
                        curr_dis = np.array(curr_dis)/curr_dis[0]
                        for each in curr_dis:
                            curr_dis_.append(each)
                        curr_dis_.append(curr_dis[0]-curr_dis[1]/curr_dis[0])
                        for i in range(2, 6):
                            curr_dis_.append(curr_dis[i]/(curr_dis[0]-curr_dis[1]))
                    elif (curr_dis[0] != 0) & (curr_dis[0]==curr_dis[1]):
                        curr_dis = np.array(curr_dis)/curr_dis[0]
                        curr_dis_ = np.zeros(11)
                        for idx, val in enumerate(curr_dis):
                            curr_dis_[idx] = val
                    else:
                        curr_dis_ = np.zeros(11)#curr_dis
                    discount.append(curr_dis_)

                ds.append(discount)
                if len(u) == batch_size:
                    u = np.array(u)
                    s = np.array(s)
                    ds = np.array(ds)
                    y = np.array(y)
                    yield [u, s, ds], y

                    u, s, ds, y = [], [], [], []


    def load_pretrained_embedding(self, embedding_path, is_random=False):
        print("loading pretrained embedding...")
        embedding_size = None
        print(embedding_path)
        #return None, 100
        with open(embedding_path, "rb") as fin:
            data = {}
            emb = pl.load(fin)
            for token in emb.keys():
                if token not in self.vocab.token2idx:
                    continue
                if embedding_size is None:
                    embedding_size = len(emb[token])
                data[token] = [float(t) for t in emb[token]]

            print(self.vocab.token_size, embedding_size)

            embedding_matrix = np.random.uniform(-0.25, 0.25, (self.vocab.token_size, embedding_size))
            for i, token in enumerate(self.vocab.token2idx.keys()):
                if token not in data:
                    continue
                embedding_matrix[i] = data[token]

            print("oov", self.vocab.token_size - len(data))
            print("embedding matrix shape", embedding_matrix.shape)
            print("embedding size", embedding_size)

            return embedding_matrix, embedding_size


    @property
    def steps_per_epoch(self):
        return self.train_steps