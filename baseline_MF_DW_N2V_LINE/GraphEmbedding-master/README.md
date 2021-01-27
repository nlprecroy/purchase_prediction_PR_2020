# GraphEmbedding

Author: Zongxi Li zongxili2@gmail.com

Credit:

The reproduce of baseline models references the codes from WeiChen Shen wcshen1994@163.com

https://github.com/shenweichen/GraphEmbedding

# Method


|   Model   | Paper                                                                                                                      |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- |
| DeepWalk  | [KDD 2014][DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | 
|   LINE    | [WWW 2015][LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)                          | 
| Node2Vec  | [KDD 2016][node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) | 

# How to run examples
1. make sure you have installed `tensorflow` or `tensorflow-gpu` on your local machine. 
2. run following commands, DeepWalk for example:
```bash
python setup.py install
cd examples
python deepwalk_jd.py
```
# Usage
The design and implementation follows simple principles(**graph in,embedding out**) as much as possible.
## Input format
we use `networkx`to create graphs.The input of networkx graph is as follows:
`node1 node2 <edge_weight>`

## DeepWalk

```python
G = nx.read_edgelist('../data/jd/JD_edgelist_pretrain_sigmoid.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',float)])# Read graph # float for weighted edge, int for binary edge

model = DeepWalk(G,walk_length=10,num_walks=80,workers=1)#init model
model.train(window_size=5,iter=3)# train model
embeddings = model.get_embeddings()# get embedding vectors
```

## LINE

```python
G = nx.read_edgelist('../data/jd/JD_edgelist_pretrain_sigmoid.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',float)])#read graph # float for weighted edge, int for binary edge

model = LINE(G,embedding_size=200,order='second') #init model,order can be ['first','second','all']
model.train(batch_size=1024,epochs=50,verbose=2)# train model
embeddings = model.get_embeddings()# get embedding vectors
```
## Node2Vec
```python
G=nx.read_edgelist('../data/jd/JD_edgelist_pretrain_sigmoid.txt',
                        create_using = nx.DiGraph(), nodetype = None, data = [('weight', float)])#read graph # float for weighted edge, int for binary edge

model = Node2Vec(G, walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 1)#init model
model.train(window_size = 5, iter = 3)# train model
embeddings = model.get_embeddings()# get embedding vectors
```


## MF
```python
G = nx.read_edgelist('../data/jd/JD_edgelist_pretrain_sigmoid.txt',
                 create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)]) #read graph # float for weighted edge, int for binary edge


model = MatFac(G, embedding_size=100, order='first')
model.train(batch_size=1024, epochs=5, verbose=2)
embeddings = model.get_embeddings()
```

Put the embedding to the following folder:
'Purchase_Pred_Stage_2/pretrained_embedding'
