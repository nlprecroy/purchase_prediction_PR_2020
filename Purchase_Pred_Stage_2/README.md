# Graph Application

__author__ = 'zongxi li'
__email__ = 'zongxili2@gmail.com'


## Installation

```
cd Purchase_Pred_Stage_2
pip install -e .
```

## Structure
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
|-- results_history.txt
|-- setup.cfg
|-- setup.py
|-- README.md

## Usage

### configure file

json

```json
{
  "model": "xxx",  # [purchase]
  "batch_size": 100,  # batch_size
  "epoch": 20,
  "data_path": "/path/to/[not split dataset for cross validation]",
  "log_path": "/path/to/log.txt",
  "cv_fold": 10,  # for not split dataset
  "split_round": 10,  # how many rounds to train for split data
}

## TRAIN MODEL
$ python -m graph_model.train graph_application/jd_graph/config.json
