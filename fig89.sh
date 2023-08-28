#!/bin/bash
for pred in "analytical" "dnn" "both" 
do
    python run_search.py --workload bert --predictor $pred --dataset_path data/firesim_training_data/firesim_results.csv --plot_only
done
