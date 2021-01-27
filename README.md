# purchase_prediction_PR
This is the source code of Pattern Recognition paper: Towards Purchase Prediction: A Transaction-based Setting and A Graph-based Method Leveraging Price Information.

# GraphEmbedding

Author: Zongxi Li zongxili2@gmail.com

# Structure
<pre>
| PR-SI Code
|-- Node_Emb_Stage_1  (Contains baseline code and edge list for generating user/item embedding)
|-- Purchase_Pred_Stage_2  (Contains purchase prediction code and processed data)
</pre>

# Step 1: Node embedding
We provide the code of node embedding methods utilized in the paper: MF, DeepWalk, Node2Vec, LINE, and LightGCN.

We provide the edge list (both weighted and binary) in the './data/jd' folder. You can produce user/item embedding using different baseline models by excuting corresponding files (see 'README.md' file under each directory).

Please put the generated embedding file into './Purchase_Pred_Stage_2/pretrained_embedding' for using in the prediction task.

# Step 2: Purchase Prediction
<pre>
| ./Purchase Prediction
|-- _ graph_application
|   |-- _ jd_graph
|       |-- config.json
|       |-- data.jsonl
|-- _ graph_model
|   |-- _ model
|       |-- _init_.py
|       |-- base_model.py
|       |-- metrics.py
|       |-- purchase.py
|   |-- _init_.py
|   |-- config.py
|   |-- dataloader.py
|   |-- train.py
|-- _pretrained_embedding
|   |-- pretrained.pkl
|-- _results
|   |-- results.txt
|-- setup.cfg
|-- setup.py
|-- README.md
</pre>

We provide the code of the proposed prediction model.

You can modify configuration (i.e., batch size, latent dimensionality, pretrained embedding) in 
<pre>
./Purchase_Pred_Stage_2/graph_application/jd_graph/config.json
</pre>
