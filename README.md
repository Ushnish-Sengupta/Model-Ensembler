# Ensembling geophysical models using Bayesian Neural Networks
This repository contains the code and pretrained models accompanying the paper "Ensembling geophysical models using Bayesian neural networks". This code has the following dependencies: python >=3.6, tensorflow-gpu == 1.15, matplotlib == 3.2.1, numpy == 1.18.5, basemap == 0.1.

The results of the toy problem may be reproduced by running Toy problem/Synthetic.ipynb. 

The models for the ozone data can be trained using Ozone/Training.py. Ozone/Evaluation.ipynb loads the pre-trained neural network files in the Ozone/Pretrained folder, produces the plots from the paper and evaluates our ensembler against the baselines.

The following table summarizes the prediction RMSEs in Dobson Units on subsets of the validation dataset.


| Method        | Temporal extrapolation | Missing North Pole | Missing South Pole | Missing Tropics | Satellite Tracks | Small Features |
| ------------- | ---------------------- | ------------------ | ------------------ | --------------- | ---------------- | -------------- |
| BayNNE        | #4.4# | 4.7 | 6.6 | 2.7 | 2.1 | 3.2 | 
| Bilinear\*     |  |  |  | 31.2 | 1.7 | 3.4 |
| Spatiotemporal Kriging\*|  |  |  | 7.0 | 2.2 | 3.4 |
| Multi-model mean | 15.7 | 8.8 | 30.5 | 9.8 | 9.2 | 16.4 |
| Weighted mean | 8.7 | 12.3 | 22.1 | 8.2 | 8.5 | 10.2 |
| Spatially weighted mean | 9.8 | 6.6 | 19.6 | 5.5 | 5.2 | 10.0 |

\*Used for interpolation only
