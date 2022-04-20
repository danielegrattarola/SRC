#!/bin/bash

for dataset in Grid2d Ring Airfoil Minnesota Bunny Sensor cora citeseer pubmed
do
    source run.sh $dataset
done