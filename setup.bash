#!/bin/bash
echo "[params]
n_value = 100
alpha_value = 2.5
beta_value = 2.5
balance_value = 3
d_value = 2
lambda_decay = 0.6
random_seed = 200
[analysis]
filename = graph.json" > params.ini

python3 analysis.py


