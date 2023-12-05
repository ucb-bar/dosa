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
    for i in range(len(layers)):
        reads, updates, writes = mapping_utils.accesses_from_mapping(denormed_full_mapping[i], layers[i], with_cache=with_cache)
        r = reads[0][0]
        if isinstance(r, int):
            r = torch.tensor(r, dtype=torch.float32)
        layer_accesses = r.unsqueeze(-1)
        for mem_lvl in range(len(relevant)):
            # this_lvl_accesses = None
            this_lvl_reads = None
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
                        this_lvl_writes = u+w
                    else:
                        # this_lvl_accesses = this_lvl_accesses + r + u + w
                        this_lvl_reads = this_lvl_reads + r
                        this_lvl_writes = this_lvl_writes + u + w
            layer_accesses = torch.cat((layer_accesses, this_lvl_reads+this_lvl_writes))
        layer_accesses = layer_accesses.unsqueeze(0) # [[x, y, z, ...]]
        if relevant_accesses is None:
            relevant_accesses = layer_accesses
        else:
            relevant_accesses = torch.cat((relevant_accesses, layer_accesses), dim=0)
    relevant_accesses = torch.clamp(relevant_accesses, 1)
    return relevant_accesses

def predict_energy(output_dir, dla_dataset):
    # get coeffs per mem level
    energy_predictor = energy_model.EnergyModel(output_dir, 3)
    energy_predictor.train(dla_dataset, valid_data=None)

    # parse mappings, layer sizes, and hw sizes
    prob_keys = utils.keys_by_type(dla_dataset.df, "prob")
    mappings = []
    layers = []
    coeffs = []
    for i, row in dla_dataset.df.iterrows():
        targets, arch_feats, prob_feats, map_feats = dla_dataset[i]
        coeff = energy_predictor.predict_coeff(arch_feats)
        coeffs.append(coeff.tolist())
        layer = eval.parse_prob(output_dir, prob_keys, prob_feats)
        layers.append(layer)
        # mapping = mapping_utils.process_mapping(row["mapping.mapping"], layer.shape)
        mapping = map_feats.tolist()
        mappings.append(mapping)
    denormed_full_mapping = torch.tensor(mappings)
    coeffs = torch.tensor(coeffs)
    
    # compute accesses per mapping
    relevant_accesses = get_accesses(denormed_full_mapping, layers)
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
    return energy_pred

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
    pred = predict_energy(args.output_dir, dla_dataset)
    logger.info(pred)

    # output_dir = DATASET_ROOT_PATH.parent / "output_dir_calibration"
    # arch_name = "gemmini"
    # workload = "resnet50"
    # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/data/gemmini_resnet50_defaultarch_best10000map_3_29_23/dataset.csv")
    # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/output_dir_mapping_driven_hw_search_model_only/gd_results_bert_0-2023-07-07--04-18-44-5WFUH5VMPBRHOXKV.csv")
    # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/data/firesim_6_14_23/dataset_firesim_6_14_23.csv")
    # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/output_dir_mapping_driven_hw_search_artifact/gd_results_resnet50_0-2023-08-01--23-51-27-S4EF7CEOICTA2VKJ.csv")
    # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/output_dir_mapping_driven_hw_search_artifact_2/gd_results_resnet50_0-2023-08-04--12-04-16-YOBALT6QVE5QT0YY.csv")
    # # dataset_path = pathlib.Path("/scratch/charleshong/dla-dataset/data/gemmini_allnets_dummyarch_1000map_4_10_23/firesim_results.csv")
    # # find_best_bandwidths(output_dir, arch_name, dataset_path)
    # # eval_layers(output_dir, arch_name, dataset_path)
    # eval_auto_tile(output_dir, arch_name, dataset_path, workload)
