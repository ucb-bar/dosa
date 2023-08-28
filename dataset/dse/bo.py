import pathlib
import traceback
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import DATASET_ROOT_PATH
from dataset.common import logger, utils, mapping_utils
from dataset.hw import init_hw_config, HardwareConfig, GemminiConfig
from dataset.workloads import Prob
from dataset.dse import ga

import sklearn.exceptions
import sklearn.inspection as insp
import scipy.stats as stats

import pandas as pd

import sklearn.gaussian_process as gp
import sklearn.preprocessing as pp

import multiprocessing as mp


class GaussianProcessOptimizer:
    def __init__(self, arch_name: str, output_dir: pathlib.Path, prob: Prob, l: float = 1.0):
        self.arch_name = arch_name
        self.output_dir = output_dir
        self.prob = prob
        self.train_X = []
        self.train_y = []
        self.scale_X = pp.StandardScaler()
        self.scale_y = pp.StandardScaler()
        self.model = gp.GaussianProcessRegressor()
        self.predictor = None
        self.l = l

    @classmethod
    def save_arch_config(cls, arch_config: HardwareConfig):
        cls.saved_arch_configs.append(arch_config)

    @classmethod
    def get_saved_arch_configs(cls):
        return cls.saved_arch_configs

    @classmethod
    def reset_saved_arch_configs(cls):
        cls.saved_arch_configs = []

    def fit(self, X, y):
        self.trained_X = X.copy()
        self.trained_y = y.copy().reshape(len(y), 1)
        train_X = self.scale_X.fit_transform(self.trained_X)
        train_y = self.scale_y.fit_transform(self.trained_y)
        self.predictor = self.model.fit(train_X, train_y)

    def predict(self, X):
        X = self.scale_X.transform(X)
        preds, stds = self.predictor.predict(X, return_std=True)
        return np.argsort(preds - self.l * stds)
    
    def predict_without_scale(self, X):
        X = self.scale_X.transform(X)
        preds, stds = self.predictor.predict(X, return_std=True)
        return preds, stds
    

# Tools
def get_layers(workload="conv_test"):
    # get the layer definitions associated with a certain workload, e.g. resnet50, alexnet, etc.
    base_workload_path = DATASET_ROOT_PATH / "workloads"
    workload_path = base_workload_path / workload

    layers = []
    unique_layers = utils.parse_yaml(workload_path / 'unique_layers.yaml')
    for unique_layer in unique_layers:
        layer_path = workload_path / (unique_layer+'.yaml')
        layer_path = layer_path.resolve()
        layers.append(Prob(layer_path))
    
    # get the number of times each layer appears in the workload
    try:
        layer_count_dict = utils.parse_yaml(workload_path / 'layer_count.yaml')
        counts = [layer_count_dict[prob.config_str()]["count"] for prob in layers]
    except:
        logger.warning("Couldn't find layer count, using default layer counts")
        counts = [1 for prob in layers]

    return layers, counts

def generate_random_mapping(size: int, prob: Prob, arch_config: HardwareConfig):
    mappings = arch_config.run_random_mappings(prob, size)
    return mappings

def get_mappings_array(mappings, prob):
    X = []
    for mapping in mappings:
        arr = mapping_utils.process_mapping(mapping['mapping.mapping'], prob.shape)
        X.append(arr)
    return X

def get_energy_array(mappings, prob):
    y = []
    for mapping in mappings:
        ener = mapping['target.energy'] * mapping['target.cycle']
        y.append(ener)
    return np.array(y)

def evaluate_mapping(prob: Prob, mapping: list, output_dir, hardward_config: str = DATASET_ROOT_PATH/'dse'/'output'):
    layers = [prob] * len(mapping)
    return ga.eval_mappings(mapping, layers, output_dir, metric="edp")

def get_best_mapping(mappings):
    best_mapping = mappings[0]
    best_energy = best_mapping['target.energy'] * best_mapping['target.cycle']
    for mapping in mappings:
        energy = mapping['target.energy'] * mapping['target.cycle']
        if energy < best_energy:
            best_energy = energy
            best_mapping = mapping
    return best_mapping

def all_hardware_config():
    pe_multiplier = range(2, 129, 2)
    buf_multipliers = [
        range(8, 1025, 8),
        range(8, 1025, 8),
    ]
    configs = []
    for pe in pe_multiplier:
        for buf in buf_multipliers[0]:
            for buf2 in buf_multipliers[1]:
                hw_config = [
                    pe,
                    buf,
                    buf2
                ]
                configs.append(hw_config)
    return configs

def random_hardware_config(size: int, output_dir: pathlib.Path = DATASET_ROOT_PATH/'dse'/'output'):
    configs = []
    for _ in range(size):
        pe_multiplier = random.choice(range(2, 129, 2))
        buf1 = random.choice(range(8, 1025, 8))
        buf2 = random.choice(range(8, 1025, 8))
        hw_config = [pe_multiplier, buf1, buf2]
        configs.append(hw_config)
    return configs



# Optimization Methods
def sw_optimize(prob: Prob, hardware_arc: HardwareConfig, train_size: int, train_trial:int, test_size: int=10000, output_dir: pathlib.Path = DATASET_ROOT_PATH/'dse'/'output'):
    # Generate random mappings
    train_mappings = generate_random_mapping(train_size, prob, hardware_arc)
    # Get X and y
    train_mappings = train_mappings[:train_size]
    X = get_mappings_array(train_mappings, prob)
    y = get_energy_array(train_mappings, prob)
    # Fit model
    gp = GaussianProcessOptimizer('gemmini', output_dir, prob)
    gp.fit(X, y)
    # Predict
    test_mappings = generate_random_mapping(test_size, prob, hardware_arc)
    X = get_mappings_array(test_mappings, prob)
    preds = gp.predict(X)
    # Return best mapping
    test_y = get_energy_array(test_mappings, prob)

    best_n = test_y[preds[:train_trial]]
    best_n_mapping = test_mappings[preds[np.argmin(best_n)]]
    return best_n_mapping

def layers_optimize(probs: list[Prob], layers_counts: list[int], hardware_arc: HardwareConfig, sw_train_size: int, sw_trail_size: int=10, sw_batch_size: int=10000, output_dir: pathlib.Path = DATASET_ROOT_PATH/'dse'/'output'):
    energy_sum = 0
    cycles_sum = 0
    for prob, layer_count in zip(probs, layers_counts):
        best_mapping = sw_optimize(prob, hardware_arc, sw_train_size, sw_trail_size, sw_batch_size, output_dir)
        energy_sum += best_mapping['target.energy'] * layer_count
        cycles_sum += best_mapping['target.cycle'] * layer_count
    return energy_sum * cycles_sum


def hw_optimize(prob: list[Prob], layer_counts: list[int], hw_train_size: int, sw_train_size: int, hw_trail_size:int, sw_trail_size:int, sw_test_size:int, output_dir: pathlib.Path = DATASET_ROOT_PATH/'dse'/'output'):
    # Generate all possible mappings
    train_hardware_configs = random_hardware_config(hw_train_size, output_dir)
    train_y = []
    for hardware_arc in train_hardware_configs:
        # Generate random mappings
        config = GemminiConfig(hardware_arc, output_dir)
        edp = layers_optimize(prob, layer_counts, config, sw_train_size, sw_trail_size, sw_test_size, output_dir)
        train_y.append(edp)
    
    t_y = train_y
    train_y = np.array(train_y)

    model = GaussianProcessOptimizer('gemmini', output_dir, prob)
    model.fit(train_hardware_configs, train_y)
    # Predict
    test_hardware_configs = all_hardware_config()
    preds = model.predict(test_hardware_configs)
    # Return best mapping
    best_n_config = np.argsort(preds)[:hw_trail_size]
    best_edp = float('inf')
    best_config = None
    for n in best_n_config:
        hardware_arc = test_hardware_configs[n]
        config = GemminiConfig(hardware_arc, output_dir)
        edp = layers_optimize(prob, layer_counts, config, sw_train_size, sw_trail_size, sw_test_size, output_dir)
        if edp < best_edp:
            best_edp = edp
            best_config = hardware_arc

    return train_hardware_configs, t_y, best_config, best_edp



# Fixed hardware and software trial size
hw_trial_size = 10
sw_trial_size = 10
sw_test_size = 1000
hw_training_size = 100
sw_training_size = 100



def main(w):
    probs, layers_counts = get_layers(w)
    training_configs, training_val, best_config, best_edp = hw_optimize(probs, layers_counts, hw_training_size, sw_training_size, hw_trial_size, sw_trial_size, sw_test_size)
    # save the result
    table = pd.DataFrame(columns=['hw_config', 'edp'])
    table['hw_config'] = training_configs
    table['edp'] = training_val
    table.to_csv(w + "_result.csv")
    with open(w + "_best_config.txt", "w") as f:
        f.write(str(best_config))
        f.write("\n")
        f.write(str(best_edp))
        f.write("\n")


if __name__ == '__main__':
    process = []
    workload = ['resnet50']
    for w in workload:
        p = mp.Process(target=main, args=(w,))
        p.start()
        process.append(p)
    
    for p in process:
        p.join()