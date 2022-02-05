# DLSM-OGB

This is an implementation of our work "Combing Latent Space Models and Graph Neural Networks for Directed Graph Representation Learning" for OGB datasets.

## Requirements

- python 3.7.6

- tensorflow 2.2.0

- ogb 1.3.2

## Examples

### Link Prediction

```
python train.py --model dlsm --dataset ogbl-citation2 --link_prediction 1
```
```
python train.py --model dlsm_d --dataset ogbl-citation2 --link_prediction 1
```

### Community Detection

```
python train.py --model dlsm --dataset ogbn-arxiv --community_detection 1
```
