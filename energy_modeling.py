import pathlib
import time
import random
import math
import argparse

import torch
import pandas as pd

from dataset import DATASET_ROOT_PATH
from dataset.workloads import Prob
from dataset.common import utils, logger, mapping_utils
from dataset.dse import energy_model, eval, DlaDatasetCreator
from dataset.hw import GemminiConfig

def load_dataset(dataset_path):
    split_ratios = {"train": 1}
    dataset_kwargs = {
        "dataset_path":dataset_path, "shuffle":False, "total_samples":0, "split_ratios":split_ratios, "process_mappings":"split",
        "target_log":False, "target_norm":None, "probfeat_log":False, "probfeat_norm":None,
        "archfeat_log":False, "archfeat_norm":None, "mapfeat_log":False, "mapfeat_norm":None, "num_processes":1,
    }
    dla_dataset_creator = DlaDatasetCreator(**dataset_kwargs)
    train_data = dla_dataset_creator.get_train_data()
    return train_data

def get_accesses(denormed_full_mapping, layers, with_cache=False):
    relevant = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
    if with_cache:
        relevant.append([1,1,1])
    relevant_accesses = None
    ruw_accesses = None
    for i in range(len(layers)):
        reads, updates, writes = mapping_utils.accesses_from_mapping(denormed_full_mapping[i], layers[i], with_cache=with_cache)
        r = reads[0][0]
        if isinstance(r, int):
            r = torch.tensor(r, dtype=torch.float32)
        layer_accesses = r.unsqueeze(-1)
        ruw_layer_accesses = torch.cat((layer_accesses, torch.tensor([0]), torch.tensor([0])))
        for mem_lvl in range(len(relevant)):
            # this_lvl_accesses = None
            this_lvl_reads = None
            this_lvl_updates = None
            this_lvl_writes = None
            for tensor in range(len(relevant[mem_lvl])):
                if relevant[mem_lvl][tensor]:
                    r = reads[mem_lvl][tensor]
                    u = updates[mem_lvl][tensor]
                    w = writes[mem_lvl][tensor]
                    if isinstance(r, int):
                        r = torch.tensor(r, dtype=torch.float32)
                    if isinstance(u, int):
                        u = torch.tensor(u, dtype=torch.float32)
                    if isinstance(w, int):
                        w = torch.tensor(w, dtype=torch.float32)
                    r = r.unsqueeze(-1)
                    u = u.unsqueeze(-1)
                    w = w.unsqueeze(-1)
                    # if this_lvl_accesses is None:
                        # this_lvl_accesses = r+u+w
                    if this_lvl_reads is None:
                        this_lvl_reads = r
                        this_lvl_updates = u
                        this_lvl_writes = w
                    else:
                        # this_lvl_accesses = this_lvl_accesses + r + u + w
                        this_lvl_reads = this_lvl_reads + r
                        this_lvl_updates = this_lvl_updates + u
                        this_lvl_writes = this_lvl_writes + w
            layer_accesses = torch.cat((layer_accesses, this_lvl_reads+this_lvl_updates+this_lvl_writes))
            mem_lvl_ruw = torch.stack([this_lvl_reads[0], this_lvl_updates[0], this_lvl_writes[0]], dim=0)
            ruw_layer_accesses = torch.cat((ruw_layer_accesses, mem_lvl_ruw), dim=0)
        layer_accesses = layer_accesses.unsqueeze(0) # [[x, y, z, ...]]
        ruw_layer_accesses = ruw_layer_accesses.unsqueeze(0)
        if relevant_accesses is None:
            relevant_accesses = layer_accesses
            ruw_accesses = ruw_layer_accesses
        else:
            relevant_accesses = torch.cat((relevant_accesses, layer_accesses), dim=0)
            ruw_accesses = torch.cat((ruw_accesses, ruw_layer_accesses), dim=0)
    relevant_accesses = torch.clamp(relevant_accesses, 1)
    return relevant_accesses, ruw_accesses

def predict_energy(output_dir, dla_dataset, run_timeloop=False):
    # get coeffs per mem level
    energy_predictor = energy_model.EnergyModel(output_dir, 3)
    energy_predictor.train(dla_dataset, valid_data=None)

    # parse mappings, layer sizes, and hw sizes
    prob_keys = utils.keys_by_type(dla_dataset.df, "prob")
    mappings = []
    layers = []
    coeffs = []
    timeloop_data = {"target.cycle": [], "target.energy": [], "target.edp": [], "target.area": []}
    for i, row in dla_dataset.df.iterrows():
        targets, arch_feats, prob_feats, map_feats = dla_dataset[i]
        coeff = energy_predictor.predict_coeff(arch_feats)
        coeffs.append(coeff.tolist())
        layer = eval.parse_prob(output_dir, prob_keys, prob_feats)
        layers.append(layer)
        # mapping = mapping_utils.process_mapping(row["mapping.mapping"], layer.shape)
        mapping = map_feats.tolist()
        mappings.append(mapping)

        # uncomment if you want to run Timeloop, will be much slower
        if run_timeloop:
            arch_config = GemminiConfig(arch_feats.tolist(), output_dir)
            mapping_dict = arch_config.flat_mapping_to_dict(layer.shape, map_feats)
            timeloop_result = arch_config.run_mapping_from_dict(layer, mapping_dict)
            timeloop_data["target.cycle"].append(timeloop_result["target.cycle"])
            timeloop_data["target.energy"].append(timeloop_result["target.energy"])
            timeloop_data["target.edp"].append(timeloop_result["target.edp"])
            timeloop_data["target.area"].append(timeloop_result["target.area"])
            print(i, timeloop_result["mapping.mapping"])
    denormed_full_mapping = torch.tensor(mappings)
    coeffs = torch.tensor(coeffs)
    
    # compute accesses per mapping
    relevant_accesses, ruw_accesses = get_accesses(denormed_full_mapping, layers)
    access_keys = utils.keys_by_type(dla_dataset.df, "dse.access")
    access_means = []
    stats = dla_dataset.creator.stats
    for access_key in access_keys:
        access_mean = stats[access_key + "_max"]
        if access_mean == 0:
            access_means.append(1)
        else:
            access_means.append(access_mean)
    access_means = torch.tensor(access_means)
    normed_relevant_accesses = relevant_accesses / access_means

    # combine coeffs and accesses
    normed_energy_pred = coeffs * normed_relevant_accesses
    energy_pred = energy_predictor.denorm_energy(normed_energy_pred)
    return energy_pred, relevant_accesses, ruw_accesses, timeloop_data

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Dataset Path',
                        required=True,
                        )
    return parser

if __name__ == "__main__":
    args = construct_argparser().parse_args()
    dla_dataset = load_dataset(args.dataset_path)
    pred, accesses, accesses_split = predict_energy(args.output_dir, dla_dataset)
    logger.info(pred)
    logger.info(accesses)