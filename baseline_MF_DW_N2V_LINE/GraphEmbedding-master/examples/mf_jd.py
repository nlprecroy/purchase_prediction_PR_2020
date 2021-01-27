
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import MatFac
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import pickle as pl

def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/jd/JD_edgelist_pretrain.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("<<<<<<<<<<<<<Starting creating graph>>>>>>>>>>>>>>>")
    G = nx.read_edgelist('../data/jd/JD_edgelist_pretrain_bin.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    print("<<<<<<<<<<<<<Grpah has been created >>>>>>>>>>>>>>>")

    model = MatFac(G, embedding_size=100, order='first')
    model.train(batch_size=1024, epochs=5, verbose=2)
    embeddings = model.get_embeddings()

    print(len(embeddings))
    
    with open("pretrained_mf.pkl", "wb") as writer:
        pl.dump(embeddings, writer, pl.HIGHEST_PROTOCOL)
    


    #evaluate_embeddings(embeddings)
    #plot_embeddings(embeddings)
