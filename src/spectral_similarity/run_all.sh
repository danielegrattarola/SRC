#!/bin/bash

for dataset in Grid2d Ring Airfoil Minnesota Bunny Sensor
do
    source run.sh $dataset
done