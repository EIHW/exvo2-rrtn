# "Redundancy Reduction Twins Network: A Training framework for Multi-output Emotion Regression"
This is the official code repo of the paper [Redundancy Reduction Twins Network: A Training framework for Multi-output Emotion Regression](https://arxiv.org/abs/2206.09142).

## Installation

~~~bash
git clone https://github.com/EIHW/exvo2-rrtn.git
cd exvo2-rrtn
pip install -r requirements.txt
~~~

## Running
- Configuration file: `train_cfg.py`
- Training script: `train_btloss.py`
- you can select the different network architecture in different train scripts.

## Note

If you have different machines to train, you could modify the [get_path_prefix](https://github.com/EIHW/exvo2-rrtn/blob/d150163f878b66013ddc03c5bf8173241ac21466/train_btloss.py#L20) to make it easier to debug and training