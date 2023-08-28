import pathlib

import numpy as np
# TODO: import genetic or evolutionary library algorithm here

from dataset import DATASET_ROOT_PATH
from dataset.common import logger, utils, mapping_utils
from dataset.hw import init_hw_config
from dataset.workloads import Prob

# CHARLES - example code to show how to evaluate mappings

output_dir = pathlib.Path("output_dir")

def aggregate_rows(rows, metric) -> float:
    cycle = 0
    energy = 0
    area = 0
    for row in rows:
        cycle += row["target.cycle"]
        energy += row["target.energy"]
        area = row["target.area"]
    if metric == "edp":
        logger.debug("Real energy: %s, real latency: %s, real EDP: %s", energy, cycle, cycle*energy)
        return cycle * energy
    elif metric == "adp":
        logger.debug("Real area: %s, real latency: %s, real ADP: %s", area, cycle, cycle*area)
        return cycle * area
    elif metric == "energy":
        return energy

def eval_mappings(mappings, layers, output_dir, hw_config=[16, 256, 64], metric="edp"):
    """Evaluate set of mappings w/ Timeloop for a given hardware config, performance metric
    for Gemmini-like HW, config consists of - [PE dim, Scratchpad size (KB), Accumulator size (KB)]

    common/mapping_utils.py:process_mapping() explains the format of the mapping array
    mappings - a 2D array (array of mappings, each of which is a 1D array)
    each row of mappings is a different mapping - could be 1 mapping for each layer in the neural network

    layers should be a list of Prob objects; see workloads/prob.py

    HardwareConfig.flat_mapping_to_dict() converts array to Timeloop yaml format
    HardwareConfig.run_mapping_from_dict() runs Timeloop
    """
    output_dir = pathlib.Path(output_dir).resolve()
    arch_config = init_hw_config("gemmini", hw_config, output_dir)
    rows = []
    for i, prob in enumerate(layers):
        mapping = mappings[i]
        row = arch_config.run_mapping_from_dict(prob, arch_config.flat_mapping_to_dict(prob.shape, mapping))

        # # to run random mappings instead and return the row with the best EDP:
        # rows = arch_config.run_random_mappings(prob, 1000, return_min_fn=lambda row: row["target.energy"] * row["target.cycle"])
        # row = rows[0]
        if row == {}:
            return float("inf")
        rows.append(row)
    return aggregate_rows(rows, metric)

def generate_cosa_mappings(hw_config, layers):
    # use CoSA (one-shot heuristic-based mapper) to generate a set of mappings for a given hardware config
    arch_config = init_hw_config("gemmini", "random", output_dir)
    mappings = []
    for prob in layers:
        flat_mapping = arch_config.run_cosa(prob, run_mapping=False)
        mappings.append(flat_mapping)
    return np.array(mappings)

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
    return layers

### add code here - probably would help to use object oriented organization?
