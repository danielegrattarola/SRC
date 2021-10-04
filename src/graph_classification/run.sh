#!/bin/bash

# Model-based, fixed, dense
python run_diffpool.py --dataset $1
python run_mincut.py --dataset $1

# Model-free, adaptive, dense
python run_nmf.py --dataset $1
python run_lapool.py --dataset $1

# Model-based, adaptive, sparse
python run_topk.py --dataset $1
python run_sagpool.py --dataset $1

# Model-free, adaptive, sparse
python run_graclus.py --dataset $1
python run_ndp.py --dataset $1