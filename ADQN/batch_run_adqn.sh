#!/bin/bash

for i in {10..100..10}
do
    python run.py --random_seed=$i 
done
