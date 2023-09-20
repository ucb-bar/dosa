import pathlib
import time
import random
import math
import argparse

import torch
import pandas as pd

from dataset import DATASET_ROOT_PATH
from dataset.common import utils, logger, mapping_utils
from dataset.dse import eval, DlaDatasetCreator

def custom_comp(x, y):
    if isinstance(x, str) and isinstance(y, str):
        for i in range(len(x)):
            if i >= len(y):
                return 1
            elif x[i] == "_" and y[i] != "_":
                return -1
            elif y[i] == "_" and x[i] != "_":
                return 1
            elif x[i] < y[i]:
                return -1
            elif x[i] > y[i]:
                return 1
    return x > y

def eval_layers(output_dir, arch_name, dataset_path):
    orig_df = pd.read_csv(dataset_path)
    dataset_kwargs = {
        "dataset_path":dataset_path, "target_norm":"", "probfeat_norm":"", "archfeat_norm":"", "mapfeat_norm":"",
        "target_log":False, "probfeat_log":False, "archfeat_log":False, "mapfeat_log":False,
        "split_ratios":{"train": 1}, "process_mappings": "split", "shuffle": False,
    }
    creator = DlaDatasetCreator(**dataset_kwargs)
    train_data = creator.get_train_data()
    c_sp_idx = mapping_utils.mapping_index(4, 7, 1, "spatial", 4)
    k_sp_idx = mapping_utils.mapping_index(4, 7, 2, "spatial", 5)

    cycles = {}
    procs = []
    for i in range(len(train_data)):
        targets, arch_feats, prob_feats, map_feats = train_data[i]

        # this function assumes that the feats passed in directly correspond to the order of keys found in `dataset`
        # convert prob feats to Prob instance
        arch_feats[0] = max(map_feats[c_sp_idx], map_feats[k_sp_idx])
        prob_keys = utils.keys_by_type(train_data.df, "prob")
        prob = eval.parse_prob(output_dir, prob_keys, prob_feats)

        proc, read_bundle, arch_config = eval.eval_layer_dataset(output_dir, arch_name, train_data, 
                                                                 arch_feats, prob_feats, map_feats=map_feats, run_async=True, **dataset_kwargs)
        procs.append((proc, read_bundle, arch_config))
        # row = eval.eval_layer_dataset(output_dir, arch_name, train_data, 
        #                               arch_feats, prob_feats, map_feats=map_feats, run_async=False, **dataset_kwargs)
        # orig_df.loc[i, "target.cycle"] = row["target.cycle"]
        # cycles[prob.config_str()] = row["target.cycle"]

    running_procs = [True for _ in procs]
    cycles = {}
    while any(running_procs):
        for i in range(len(procs)):
            if not running_procs[i]:
                continue
            proc = procs[i][0]
            retcode = proc.poll()
            if retcode is not None:
                read_bundle = procs[i][1]
                prob = read_bundle[0]
                arch_config = procs[i][2]
                row = arch_config.read_run_mapping(*read_bundle)
                logger.debug("%s %s", row["target.cycle"], prob.config_str())
                cycles[prob.config_str()] = row["target.cycle"]
                running_procs[i] = False
                orig_df["target.cycle"][i] = row["target.cycle"]
        time.sleep(0.5)

    # orig_df.to_csv("/scratch/charleshong/dla-dataset/data/gemmini_allnets_dummyarch_1000map_4_10_23/firesim_results_bw.csv", index=False)
    sorted_layers = sorted(cycles.keys())#, key=functools.cmp_to_key(custom_comp))
    for k in sorted_layers:
        print(k)
    for k in sorted_layers:
        print(cycles[k])
    return cycles

def eval_auto_tile(output_dir, arch_name, dataset_path, workload):
    orig_df = pd.read_csv(dataset_path)
    dataset_kwargs = {
        "dataset_path":dataset_path, "target_norm":"", "probfeat_norm":"", "archfeat_norm":"", "mapfeat_norm":"",
        "target_log":False, "probfeat_log":False, "archfeat_log":False, "mapfeat_log":False,
        "split_ratios":{"train": 1}, "process_mappings": "split", "shuffle": False,
    }
    creator = DlaDatasetCreator(**dataset_kwargs)
    train_data = creator.get_train_data()
    c_sp_idx = mapping_utils.mapping_index(4, 7, 1, "spatial", 4)
    k_sp_idx = mapping_utils.mapping_index(4, 7, 2, "spatial", 5)

    results = {}
    tiled_min_hw = None
    for energy_type in ["auto", "tiled"]:
        groups = []
        for i in range(0, len(train_data), 16):
            groups.append(list(range(i, min(i + 16, len(train_data)))))
        for group_idxs in groups:
            procs = []
            for i in group_idxs:
                targets, arch_feats, prob_feats, map_feats = train_data[i]

                # this function assumes that the feats passed in directly correspond to the order of keys found in `dataset`
                # convert prob feats to Prob instance
                prob_keys = utils.keys_by_type(train_data.df, "prob")
                prob = eval.parse_prob(output_dir, prob_keys, prob_feats)
                if prob.config_str() not in results:
                    results[prob.config_str()] = {
                        "auto_cycle": orig_df["target.gemmini_auto_cycle"][i],
                        # "tiled_energy": orig_df["target.energy"][i],
                        "tiled_cycle": orig_df["target.gemmini_cycle"][i],
                    }
                if energy_type == "tiled":
                    tiled_min_hw = list(arch_feats)
                # if "dse.auto_tiling" not in orig_df.columns:
                #     continue

                # replace mapping with auto_tiling mapping
                reg_start_idx = mapping_utils.mapping_index(4, 7, 0, "temporal", 0)
                acc_start_idx = mapping_utils.mapping_index(4, 7, 1, "temporal", 0)
                if energy_type == "auto":
                    map_feats[14:21] = torch.tensor([2, 3, 5, 6, 1, 4, 7]) # CRSKPQN - DRAM loop ordering
                    map_feats[reg_start_idx:reg_start_idx+7] = torch.tensor([1] * 7)
                    acc_factors = torch.tensor([int(f) for f in orig_df["dse.auto_tiling"][i].split("_")])
                    acc_factors[4] = max(1, torch.div(acc_factors[4], 16, rounding_mode="trunc"))
                    acc_factors[5] = max(1, torch.div(acc_factors[5], 16, rounding_mode="trunc"))
                    map_feats[acc_start_idx:acc_start_idx+7] = acc_factors
                    map_feats[c_sp_idx] = 16
                    map_feats[k_sp_idx] = 16

                map_feats[acc_start_idx+7:acc_start_idx+14] = torch.tensor([5, 6, 1, 2, 4, 7, 3]) # PQNCRSK accumulator loop loop ordering
                map_feats[reg_start_idx+7:reg_start_idx+14] = torch.tensor([3, 4, 1, 2, 5, 6, 7]) # PQRSCKN reg loop ordering

                map_feats = mapping_utils.round_mapping(map_feats, prob, round_down=True)
                if energy_type == "auto":
                    arch_feats = [16, 128, 32]

                proc, read_bundle, arch_config = eval.eval_layer_dataset(output_dir, arch_name, train_data, 
                                                                        arch_feats, prob_feats, map_feats=map_feats, run_async=True, **dataset_kwargs)
                procs.append((proc, read_bundle, arch_config))
                # row = eval.eval_layer_dataset(output_dir, arch_name, train_data, 
                #                               arch_feats, prob_feats, map_feats=map_feats, run_async=False, **dataset_kwargs)
                # orig_df.loc[i, "target.cycle"] = row["target.cycle"]
                # cycles[prob.config_str()] = row["target.cycle"]

            running_procs = [True for _ in procs]
            cycles = {}
            while any(running_procs):
                for i in range(len(procs)):
                    if not running_procs[i]:
                        continue
                    proc = procs[i][0]
                    retcode = proc.poll()
                    if retcode is not None:
                        read_bundle = procs[i][1]
                        prob = read_bundle[0]
                        arch_config = procs[i][2]
                        row = arch_config.read_run_mapping(*read_bundle)
                        logger.debug("%s %s", row["target.energy"], prob.config_str())
                        results[prob.config_str()][f"{energy_type}_energy"] = row["target.energy"]
                        running_procs[i] = False
                        # orig_df.at[i, "target.cycle"] = row["target.cycle"]
                time.sleep(0.5)

    layer_count = get_layer_count_dict(workload)

    perf_types = ["tiled_energy", "tiled_cycle", "auto_cycle", "auto_energy"]
    # if "dse.auto_tiling" in orig_df.columns:
    #     perf_types.extend(["auto_energy"])

    # # orig_df.to_csv("/scratch/charleshong/dla-dataset/data/firesim_6_14_23/dataset_firesim_6_14_23_cycle.csv", index=False)
    sorted_layers = sorted(results.keys())#, key=functools.cmp_to_key(custom_comp))
    for k in sorted_layers:
        print(k)
    for k in sorted_layers:
        for t in perf_types:
            results[k][t] = results[k][t] * layer_count[k]
    # for t in perf_types:
    #     print(t, sum([results[k][t] for k in sorted_layers]))
    print()
    dosa_energy = sum([results[k]["tiled_energy"] for k in sorted_layers])
    dosa_cycle = sum([results[k]["tiled_cycle"] for k in sorted_layers])
    print("DOSA energy:", dosa_energy)
    print("DOSA cycle:", dosa_cycle)
    print("DOSA EDP:", dosa_energy * dosa_cycle)
    print("DOSA min HW:", ", ".join([str(float(x)) for x in tiled_min_hw]))
    print()
    auto_energy = sum([results[k]["auto_energy"] for k in sorted_layers])
    auto_cycle = sum([results[k]["auto_cycle"] for k in sorted_layers])
    print("Gemmini default energy:", auto_energy)
    print("Gemmini default cycle:", auto_cycle)
    print("Gemmini default EDP:", auto_energy * auto_cycle)

    return results

def get_layer_count_dict(workload) -> list[int]:
    base_workload_path = DATASET_ROOT_PATH / "workloads"
    workload_path = base_workload_path / workload
    layer_count_dict = utils.parse_yaml(workload_path / 'layer_count.yaml')
    layer_count_dict = {k: v["count"] for k, v in layer_count_dict.items()}
    return layer_count_dict

def get_diff(orig_cycles, cycles, diff_type):
    if diff_type == "mean":
        sum = 1
        for k in cycles:
            sum = sum + abs(orig_cycles[k] - cycles[k])
        diff = sum / len(cycles)
    
    return diff

def find_best_bandwidths(output_dir, arch_name, dataset_path):
    dataset_kwargs = {
        "dataset_path":dataset_path, "target_norm":"", "probfeat_norm":"", "archfeat_norm":"", "mapfeat_norm":"",
        "target_log":False, "probfeat_log":False, "archfeat_log":False, "mapfeat_log":False,
        "split_ratios":{"train": 1}, "process_mappings": "split",
    }
    creator = DlaDatasetCreator(**dataset_kwargs)
    train_data = creator.get_train_data()
    orig_cycles = {
        "1_1_14_14_1024_256_1_1_1_1_1":	555109,
        "1_1_14_14_1024_512_1_1_1_1_1":	1592115,
        "1_1_14_14_256_1024_1_1_1_1_1":	674019,
        "1_1_14_14_512_1024_1_2_2_1_1":	1726466,
        "1_1_1_1_2048_1000_1_1_1_1_1":	3263447,
        "1_1_28_28_128_512_1_1_1_1_1":	464472,
        "1_1_28_28_256_512_1_2_2_1_1":	2384497,
        "1_1_28_28_512_128_1_1_1_1_1":	410530,
        "1_1_28_28_512_256_1_1_1_1_1":	765626,
        "1_1_56_56_256_128_1_1_1_1_1":	1091395,
        "1_1_56_56_256_64_1_1_1_1_1":	506834,
        "1_1_56_56_64_256_1_1_1_1_1":	646522,
        "1_1_56_56_64_64_1_1_1_1_1":	184822,
        "1_1_7_7_1024_2048_1_2_2_1_1":	3412145,
        "1_1_7_7_2048_512_1_1_1_1_1":	2013871,
        "1_1_7_7_512_2048_1_1_1_1_1":	2039111,
        "3_3_14_14_256_256_1_1_1_1_1":	1177653,
        "3_3_14_14_256_256_1_2_2_1_1":	1133786,
        "3_3_28_28_128_128_1_1_1_1_1":	661622,
        "3_3_28_28_128_128_1_2_2_1_1":	846896,
        "3_3_56_56_64_64_1_1_1_1_1":	1068003,
        "3_3_7_7_512_512_1_1_1_1_1":	3728186,
        "3_3_7_7_512_512_1_2_2_1_1":	5045433,
        "7_7_112_112_3_64_1_2_2_1_1":	975316,
    }

    min_diff = float("inf")
    min_diff_config = None
    min_diff_cycles = None
    for i in range(100):
        # bw_feats = [
        #     random.choice(range(1, 161, 1)) / 10,
        #     random.choice(range(1, 161, 1)) / 10,
        #     random.choice(range(1, 161, 1)) / 10,
        #     random.choice(range(1, 161, 1)) / 10,
        #     random.choice(range(1, 81, 1)) / 10,
        # ]
        bw_feats = [
            2 ** (random.choice(range(1, 401, 1)) / 100),
            2 ** (random.choice(range(1, 401, 1)) / 100),
            2 ** (random.choice(range(1, 401, 1)) / 100),
            2 ** (random.choice(range(1, 401, 1)) / 100),
            2 ** (random.choice(range(1, 301, 1)) / 100),
        ]
        procs = []
        for i in range(len(train_data)):
            targets, arch_feats, prob_feats, map_feats = train_data[i]
            arch_feats = arch_feats.tolist() + bw_feats

            # this function assumes that the feats passed in directly correspond to the order of keys found in `dataset`
            # convert prob feats to Prob instance
            prob_keys = utils.keys_by_type(train_data.df, "prob")
            prob = eval.parse_prob(output_dir, prob_keys, prob_feats)

            proc, read_bundle, arch_config = eval.eval_layer_dataset(output_dir, arch_name, train_data, 
                                                                    arch_feats, prob_feats, map_feats=map_feats, run_async=True, **dataset_kwargs)
            procs.append((proc, read_bundle, arch_config))

        running_procs = [True for _ in procs]
        cycles = {}
        failed_iter = False
        while any(running_procs):
            for i in range(len(procs)):
                if not running_procs[i]:
                    continue
                proc = procs[i][0]
                retcode = proc.poll()
                if retcode is not None:
                    read_bundle = procs[i][1]
                    prob = read_bundle[0]
                    arch_config = procs[i][2]
                    try:
                        row = arch_config.read_run_mapping(*read_bundle)
                        logger.debug("%s %s", row["target.cycle"], prob.config_str())
                        cycles[prob.config_str()] = row["target.cycle"]
                    except:
                        failed_iter = True
                    running_procs[i] = False
            time.sleep(0.5)
        
        if failed_iter:
            continue
        diff = get_diff(orig_cycles, cycles, "mean")
        print(i, bw_feats, diff)
        if diff < min_diff:
            min_diff = diff
            min_diff_config = bw_feats
            min_diff_cycles = cycles
    
    sorted_layers = sorted(min_diff_cycles.keys())#, key=functools.cmp_to_key(custom_comp))
    for k in sorted_layers:
        print(k)
    for k in sorted_layers:
        print(min_diff_cycles[k])

    print(min_diff_config)
    print(min_diff)

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('-wl',
                        '--workload',
                        type=str,
                        help='<Required> Name of workload directory.',
                        required=True,
                        )
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Dataset Path',
                        required=True,
                        )
    return parser

if __name__ == "__main__":
    args = construct_argparser().parse_args()
    eval_auto_tile(args.output_dir, "gemmini", args.dataset_path, args.workload)

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
