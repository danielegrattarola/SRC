#!/bin/bash

python run_sum.py --dataset $1
python run_avg.py --dataset $1
python run_max.py --dataset $1
python run_attn.py --dataset $1
python run_sort.py --dataset $1
