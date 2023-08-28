#!/bin/bash
tar -xzf data/timeloop_dataset/dataset.csv.tar.gz -C data/timeloop_dataset/
python run_search.py --workload resnet50 --predictor analytical --plot_only --dataset_path data/timeloop_dataset/dataset.csv
