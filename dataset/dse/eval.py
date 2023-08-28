import pathlib
import math
import os

import numpy as np
import pandas as pd

from dataset.common import logger, utils, mapping_utils
from dataset.hw import GemminiConfig, SimbaConfig, init_hw_config
from dataset.dse import DlaDatasetCreator, DlaDataset
from dataset.workloads import Prob

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
        if row == {}:
            return float("inf")
        rows.append(row)
    return aggregate_rows(rows, metric)

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

def eval_network(output_dir: pathlib.Path,
                 arch_name: str,
                 arch_config: list,
                 network_path: pathlib.Path,
                 mappings: list = None,
                 num_random_mappings: int = 1000,
                 obj: str = "edp") -> dict:
    """Evaluation a hardware config on a full neural network

    Args:
        mappings: a list of one mapping per unique layer (if random mapper not used)
    TODO: implement
    """
    workload_path = network_path
    unique_layers = utils.parse_yaml(workload_path / 'unique_layers.yaml')
    result_rows = []
    for layer_idx, unique_layer in enumerate(unique_layers):
        layer_path = workload_path / (unique_layer+'.yaml')
        layer_path = layer_path.resolve()
        prob = Prob(layer_path)

        # Get mapping for this layer, if it exists
        mapping = None
        if mappings is not None:
            mapping = mappings[layer_idx]

        result_row = eval_layer(output_dir,
                                arch_name,
                                arch_config,
                                prob,
                                mapping,
                                num_random_mappings = num_random_mappings,
                                obj = obj)
        result_rows.append(result_row)
    
    total_cycle = 0
    total_energy = 0
    for layer_idx, row in result_rows:
        total_cycle += row["target.cycle"]
        total_energy += row["target.energy"]
    area = result_rows[0]["target.area"]

    network_result = {
        "target.cycle": total_cycle, # cycles
        "target.energy": total_energy, # uJ
        "target.edp": total_cycle * total_energy * 1e-6, # cycles * J
        "target.area": area, # mm^2
    }
    return network_result, result_rows

def parse_prob(output_dir, prob_keys, prob_feats):
    # convert prob feats to Prob instance
    prob_dict = dict()
    prob_shape = ""
    for idx, key in enumerate(prob_keys):
        if key == "prob.shape_cls":
            prob_dict["shape"] = utils.prob_class_to_shape(round(float(prob_feats[idx])))
            prob_shape = prob_dict["shape"]
        else:
            # 5: is to remove "prob." from start of string
            prob_dict[key[5:]] = round(float(prob_feats[idx]))
    prob_dict = {"problem": prob_dict}
    # prob_dir = output_dir / "prob"
    # prob_dir.mkdir(exist_ok=True)
    # filename = utils.unique_filename("yaml", prefix=prob_shape)
    # prob_path = pathlib.Path(prob_dir / filename)
    # utils.store_yaml(prob_path, prob_dict)
    prob = Prob(prob_dict)
    # os.remove(prob_path)
    return prob

def eval_layer_dataset(output_dir: pathlib.Path, arch_name: str, dataset: DlaDataset, arch_feats: np.ndarray, prob_feats: np.ndarray, map_feats: np.ndarray = None, **dataset_kwargs):
    """
    dataset is used to get column headers and normalization/log stats
    """
    logger.debug("Evaluating arch %s %s, prob %s, map %s, result stored to %s", arch_name, arch_feats, prob_feats, map_feats, output_dir)
    output_dir = pathlib.Path(output_dir).resolve()

    arch_feats = dataset.denorm("arch", arch_feats)
    prob_feats = dataset.denorm("prob", prob_feats)
    map_feats = dataset.denorm("mapping", map_feats)

    # this function assumes that the feats passed in directly correspond to the order of keys found in `dataset`
    prob_keys = utils.keys_by_type(dataset.df, "prob")
    map_keys = utils.keys_by_type(dataset.df, "mapping")

    # convert prob feats to Prob instance
    prob = parse_prob(output_dir, prob_keys, prob_feats)

    # convert mapping feats to flat mapping
    num_mem_lvls = utils.num_mem_lvls(arch_name)
    num_dims = len(prob.prob_name_idx_dict)
    mapping = None
    if map_feats is not None:
        map_dict = {map_keys[i]: float(map_feats[i]) for i in range(len(map_keys))}
        mapping = [0] * num_mem_lvls * num_dims * 3
        dims = utils.get_prob_dims(prob.shape)
        num_dims = len(dims)
        mapping = []
        for mem_lvl in reversed(range(num_mem_lvls)): # Count down mem lvls
            for factor_type in ["spatial", "temporal", "perm"]:
                for dim in dims:
                    key = f"mapping.{factor_type}_L{mem_lvl}_{dim}"
                    mapping.append(map_dict[key])
        mapping = mapping_utils.round_mapping(mapping, prob)

    # convert arch feats to HardwareConfig - done in DLA dataset class
    hw_config = [float(feat) for feat in arch_feats]

    return eval_layer(output_dir, arch_name, hw_config, prob, mapping, **dataset_kwargs)


def eval_layer(output_dir: pathlib.Path,
               arch_name: str,
               hw_config: list,
               prob: Prob,
               mapping: list = None,
               num_random_mappings: int = 1000,
               run_cosa: bool = False,
               obj: str = "edp",
               run_async: bool = False,
               **dataset_kwargs) -> dict:
    """Evaluate a hardware config on a layer

    Args:
        arch_name (str): [gemmini, simba]
        arch (list): Initializer for hw config
            gemmini: [pe_dim, sp_size (KB), acc_size (KB)]
        layer_path (pathlib.Path): Path to layer to eval on
        mapping (list): Custom mapping. If None, run num_random_mappings
            random mappings.
        num_random_mappings (int): Number of random mappings to run.
        obj (str): This function returns all perf metrics, but if random mappings are run,
            it picks the mapping that optimizes this objective. [cycle, energy, edp, area, cycle/area]
    """
    if arch_name == "gemmini":
        arch_config = GemminiConfig(hw_config, output_dir)
    elif arch_name == "simba":
        arch_config = SimbaConfig(hw_config, output_dir)
    else:
        logger.error("arch_name %s not valid for evaluation", arch_name)

    # Each case should return a row dict
    if mapping is not None:
        mapping_dict = arch_config.flat_mapping_to_dict(prob.shape, mapping)
        row_or_proc = arch_config.run_mapping_from_dict(prob, mapping_dict, run_async=run_async)
    else:
        if run_cosa:
            run_result_rows = arch_config.run_cosa(prob)
        else:
            run_result_rows = arch_config.run_random_mappings(prob, num_random_mappings)
        df = pd.DataFrame(run_result_rows)

        needed_keys = ["target_log", "probfeat_log", "archfeat_log", "mapfeat_log",
                       "target_norm", "probfeat_norm", "archfeat_norm", "mapfeat_norm",
                       "stats_path"]
        needed_dataset_kwargs = {key: dataset_kwargs[key] for key in needed_keys}
        dla_dataset = DlaDatasetCreator(df,
                                        split_ratios={"train":1},
                                        total_samples=0,
                                        shuffle=False,
                                        **needed_dataset_kwargs,
                                        process_mappings="min",
                                        process_mappings_obj=obj)
        row_or_proc = dla_dataset.get_train_data().df.reset_index().iloc[0,]

    if run_async:
        return row_or_proc[0], row_or_proc[1], arch_config
    return row_or_proc # just return the whole row

    # target_keys = utils.keys_by_type(row, "target")
    # return {key: row[key] for key in target_keys}

if __name__ == "__main__":
    from dataset import DATASET_ROOT_PATH
    _DATA_DIR = DATASET_ROOT_PATH.parent.resolve() / "data"

    dataset_path = _DATA_DIR / "gemmini_resnet50_defaultarch_1000map_11_14_22" / "dataset.csv"
    output_dir = pathlib.Path("output_dir_eval_test")
    dataset_kwargs = {
        "dataset_path":dataset_path,
        "split_ratios":{"train":0.75, "valid":0.1, "test": 0.15},
        "total_samples":100,
        "shuffle":True,
        "target_log":False, "target_norm":"",
        "probfeat_log":False, "probfeat_norm":"",
        "archfeat_log":False, "archfeat_norm":"",
        "mapfeat_log":False, "mapfeat_norm":"",
        "process_mappings":"split",
        "process_mappings_obj":"edp",
    }
    dla_dataset_creator = DlaDatasetCreator(**dataset_kwargs)
    dataset_kwargs["stats_path"] = dla_dataset_creator.stats_path
    train_dataset = dla_dataset_creator.get_train_data()
    targets, arch_feats, prob_feats, map_feats = train_dataset[0]
    print(train_dataset[0])
    row = eval_layer_dataset(output_dir, "gemmini", train_dataset, arch_feats, prob_feats, map_feats, **dataset_kwargs)
