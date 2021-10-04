#!/usr/bin/env bash
python run_diffpool.py --F 1
python run_topk.py --F 1
python run_lapool.py --F 1

python run_diffpool.py --F 10
python run_topk.py --F 10
python run_lapool.py --F 10

python run_diffpool.py --F 100
python run_topk.py --F 100
python run_lapool.py --F 100

python run_diffpool.py --F 1000
python run_topk.py --F 1000
python run_lapool.py --F 1000
