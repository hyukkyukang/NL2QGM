# NL2QGM
Natural Language to Query Graph Model (QGM)

## Environment Setup

* `Ubuntu 16.04`
* `Python3.6`
* `Pytorch 1.4.0` or higher

Install Python dependency: `pip install -r requirements.txt`

## Running Code

#### Data preparation

* Download [Glove Embedding](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put `glove.42B.300d` under `./data/` directory  
* Follow instructions under `./qgm/` directory

#### Training

Run `train.py` to train the model.

##### Arguments

* `--train_config:` Path to jsonnet file that contains training settings  
* `--cuda:` GPU number  

#### Testing

Run `eval.py` to evaluate the model

##### Arguments

* `--train_config:` Path to jsonnet file that contains training settings  
* `--cuda:` GPU number  
* `--load_model` Path to trained model  

#### tensorboard

* After running train.py, run `tensorboard --logdir ./logs --bind_all` in terminal
* check http://[ip]:6006
