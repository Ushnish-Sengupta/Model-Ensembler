# Ensembling geophysical models using Bayesian Neural Networks
This repository contains the code, dataset and pretrained models accompanying the paper "Ensembling geophysical models using Bayesian neural networks". This code has the following dependencies: python >=3.6, tensorflow version == 1.15, matplotlib == ?, numpy == 1.15.
The results of the toy model may be reproduced by running Toy data/Synthetic.ipynb.
Training the models 


| Method        | Temporal extrapolation | Missing North Pole | Missing South Pole | Missing Tropics | Satellite Tracks | Small Features |
| ------------- | ---------------------- | ------------------ | ------------------ | --------------- | ---------------- | -------------- |
| BayNNE        | Content Cell           |                    |                    |                 |                  |                | 
| Bilinear      | Content Cell           |                    |                    |                 |                  |                |
| ST Kriging    | Content Cell           |                    |                    |                 |                  |                |
| Multi-model mean | Content Cell        |                    |                    |                 |                  |                |
| Weighted mean | Content Cell           |                    |                    |                 |                  |                |
| REA           | Content Cell           |                    |                    |                 |                  |                |
