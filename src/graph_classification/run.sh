#!/bin/bash

python run_diffpool.py --dataset $1
python run_mincut.py --dataset $1

python run_nmf.py --dataset $1
python run_lapool.py --dataset $1

python run_topk.py --dataset $1
python run_sagpool.py --dataset $1

python run_graclus.py --dataset $1
python run_ndp.py --dataset $1