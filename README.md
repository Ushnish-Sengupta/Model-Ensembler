# Ensembling geophysical models using Bayesian Neural Networks
This repository contains the code and pretrained models accompanying the paper "Ensembling geophysical models using Bayesian neural networks". This code has the following dependencies: python >=3.6, tensorflow-gpu == 1.15, matplotlib == ?, numpy == ?.

The results of the toy problem may be reproduced by running Toy data/Synthetic.ipynb. 

The models for the ozone data can be trained using Training.ipynb. Evaluation.ipynb loads the pre-trained neural network files in the Ozone folder, produces the plots from the paper and evaluates our ensembler against the baselines.

The following table summarizes the prediction RMSEs on subsets of the validation dataset.


| Method        | Temporal extrapolation | Missing North Pole | Missing South Pole | Missing Tropics | Satellite Tracks | Small Features |
| ------------- | ---------------------- | ------------------ | ------------------ | --------------- | ---------------- | -------------- |
| BayNNE        | Content Cell           |                    |                    |                 |                  |                | 
| Bilinear      | Content Cell           |                    |                    |                 |                  |                |
| ST Kriging    | Content Cell           |                    |                    |                 |                  |                |
| Multi-model mean | Content Cell        |                    |                    |                 |                  |                |
| Weighted mean | Content Cell           |                    |                    |                 |                  |                |
| REA           | Content Cell           |                    |                    |                 |                  |                |
