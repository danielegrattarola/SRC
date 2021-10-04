#!/bin/bash

# Learnable, fixed, dense
python run_diffpool.py --name $1
python run_mincut.py --name $1

# Static, adaptive, dense
python run_nmf.py --name $1
python run_lapool.py --name $1

# Learnable, adaptive, sparse
python run_topk.py --name $1
python run_sagpool.py --name $1

# Static, adaptive, sparse
python run_ndp.py --name $1
python run_graclus.py --name $1