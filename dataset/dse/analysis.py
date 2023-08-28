import pathlib
import time
import traceback
import math
import random
from collections.abc import Iterable
from typing import Callable

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import scipy

from dataset import DATASET_ROOT_PATH
from dataset.common import utils, logger
from dataset.dse import DlaDatasetCreator, eval, pytorch_util
from dataset.workloads import Prob # TODO: make importing layers simpler

_DATA_DIR = DATASET_ROOT_PATH.parent.resolve() / "data"

def run_mlp(train_data, test_data, x_key_types, y_key):
    train_df = train_data.df
    
    x_keys = []
    for key_type in x_key_types:
        type_keys = utils.keys_by_type(train_df, key_type, scalar_only=True)
        x_keys.extend(type_keys)
    X_train = train_df[x_keys]
    y_train = train_df[y_key]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=utils.get_random_seed())
    test_df = test_data.df
    X_test = test_df[x_keys]
    y_test = test_df[y_key]

    pytorch_util.init_gpu(gpu_id=0)
    hidden_layer_sizes = (128, 256, 256, 32)
    mlp = pytorch_util.build_mlp(
        input_size=len(x_keys),
        output_size=1,
        n_layers=4,
        size=hidden_layer_sizes,
        activation="relu"
    )
    mlp.to(pytorch_util.device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X_train = pytorch_util.from_numpy(X_train.to_numpy())
    y_train = pytorch_util.from_numpy(y_train.to_numpy())
    train_dataset = pytorch_util.X_y_dataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20000)
    for iter in range(200):
        for X_batch, y_batch in train_data_loader:
            y_train_pred_batch = mlp(X_batch).squeeze()
            loss = loss_fn(y_train_pred_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"Finished training iter {iter}, loss {loss.item()}")
    torch.save(mlp.state_dict(), "mlp.pt")
    torch.save(optimizer.state_dict(), "mlp_opt.pt")

    X_test = pytorch_util.from_numpy(X_test.to_numpy())
    y_test = pytorch_util.from_numpy(y_test.to_numpy())
    test_dataset = pytorch_util.X_y_dataset(X_test, y_test)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20000)
    y_pred = np.array([])
    for X_batch, _ in test_data_loader:
        y_batch_pred = mlp(X_batch).squeeze()
        y_pred = np.concatenate((y_pred, pytorch_util.to_numpy(y_batch_pred)))

    # mlp = MLPRegressor(hidden_layer_sizes, random_state=utils.get_random_seed())
    # mlp.fit(X_train, y_train)
    # y_pred = mlp.predict(X_test)
    return pytorch_util.to_numpy(y_test), y_pred

def pred_perf(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str):
    output_dir = pathlib.Path(output_dir)
    min_data_creator = DlaDatasetCreator(dataset_path=dataset_path, total_samples=100, split_ratios={"train": 0.8, "test": 0.2}, process_mappings="split")
    min_data_train = min_data_creator.train_data
    min_data_test = min_data_creator.test_data
    # split_data = DlaDataset(dataset_path=dataset_path, total_samples=10000, split_ratios={"train": 1.0}, process_mappings="split")
    y_test, y_pred = run_mlp(min_data_train, min_data_test, ("prob", "mapping"), target_key)

    print(y_test, y_pred)
    mse_without_arch = mean_squared_error(y_test, y_pred)

    plt.figure()
    plt.title(f"{target_key} prediction without arch")
    plt.xlabel("real")
    plt.ylabel("pred")
    plt.scatter(y_test, y_pred)
    plt.savefig(output_dir / "pred_without_arch.png")

    y_test, y_pred = run_mlp(min_data_train, min_data_test, ("arch", "prob", "mapping"), target_key)

    plt.figure()
    plt.title(f"{target_key} prediction with arch")
    plt.xlabel("real")
    plt.ylabel("pred")
    plt.scatter(y_test, y_pred)
    plt.savefig(output_dir / "pred_with_arch.png")

    mse_with_arch = mean_squared_error(y_test, y_pred)
    logger.info("MSE without arch: %s", mse_without_arch)
    logger.info("MSE with arch: %s", mse_with_arch)
    logger.info("MSE improvement: %s%", round((1 - mse_with_arch / mse_without_arch)*100, 1))

def pred_perf_2(dataset_path):
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=192000, split_ratios={"train": 0.75, "test": 0.25}, process_mappings="split")
    train_data = dla_dataset_creator.train_data
    test_data = dla_dataset_creator.test_data
    y_test, y_pred = run_mlp(train_data, test_data, ("arch", "prob", "mapping"), "target.edp")
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info("MSE of arch/prob split: %s", mse)
    logger.info("MAPE of arch/prob split: %s", mape)

def theoretical_min_cycles(row: pd.DataFrame) -> int:
    output_size = row["prob.P"] * row["prob.Q"]
    weight_size = row["prob.R"] * row["prob.S"]
    in_channel = row["prob.C"]
    out_channel = row["prob.K"]
    batch = row["prob.N"]
    total_macs = output_size * weight_size * in_channel * out_channel * batch

    hw_macs = row["arch.pe_dim"] ** 2 # TODO: update for new gemmini repr
    min_cycles = math.ceil(total_macs / hw_macs)
    return min_cycles

def get_matching_rows(df: pd.DataFrame, matching_keys: Iterable, matching_values: Iterable) -> pd.DataFrame:
    idxs_per_key = [df[matching_keys[i]] == matching_values[i] for i in range(len(matching_values))]
    idxs = idxs_per_key[0]
    for key_idxs in idxs_per_key:
        idxs = np.bitwise_and(idxs, key_idxs)
    comp_df = df.loc[idxs]
    return comp_df

def vis_mappings(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str, num_layers: int, mappings_per_layer: int, best_part: float = 1.0, 
                 cosa_dataset_path: pathlib.Path = None, use_search_curve=False):
    """
    Visualize distribution of mappings
    """
    # first visualize mappings for one arch
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
            target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
            archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df

    cosa_df = None
    if cosa_dataset_path:
        cosa_dataset_creator = DlaDatasetCreator(dataset_path=cosa_dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
                target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
                archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
        cosa_df = cosa_dataset_creator.train_data.df

    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = arch_keys + prob_keys
    first_idxs = df.groupby(by=matching_keys, sort=False)["index"].first()
    # print(prob_keys) # ['prob.shape', 'prob.C', 'prob.Hdilation', 'prob.Hstride', 'prob.K', 'prob.N', 'prob.P', 'prob.Q', 'prob.R', 'prob.S', 'prob.Wdilation', 'prob.Wstride'] 
    # for i in range(num_layers):
    #     row_idx = mappings_per_layer*i
    for i, row_idx in enumerate(first_idxs):
        matching_values = df.loc[row_idx, matching_keys]
        comp_rows = get_matching_rows(df, matching_keys, matching_values)
        vals = comp_rows[target_key][:mappings_per_layer]
        # best_idxs = np.argpartition(vals, int(len(vals)*best_part)-1)[:int(len(vals)*best_part)]
        # vals = np.array(vals)[best_idxs]
        print(len(vals))
        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        row = df.loc[row_idx, :]
        if cosa_df is not None:
            prob_values = df.loc[row_idx][prob_keys]
            cosa_val = get_matching_rows(cosa_df, prob_keys, prob_values)[target_key].values[0]
        min_cycles = theoretical_min_cycles(row)
        for fig in range(2):
            plt.figure()
            plt.title(prob)
            if fig == 0:
                plt.ylabel("count")
                plt.xlabel(target_key)
                plt.hist(sorted(vals)[:int(len(vals)*best_part)], bins=100)
                plt.savefig(output_dir / f"mapping_hist_{i}.png", bbox_inches="tight")
            if fig == 1:
                plt.ylabel(target_key)
                plt.xlabel("mapping")
                if use_search_curve:
                    plot_vals = utils.search_curve(vals)[-int(len(vals)*best_part):]
                else:
                    plot_vals = list(reversed(sorted(vals)))[-int(len(vals)*best_part):]
                plt.plot(range(len(plot_vals)), plot_vals, label="random")
                if cosa_df is not None:
                    plt.plot(range(len(plot_vals)), [cosa_val] * len(plot_vals), label="cosa")
                if target_key == "target.cycle":
                    plt.plot(range(len(plot_vals)), [min_cycles] * len(plot_vals), label="theoretical min")
                plt.legend()
                plt.savefig(output_dir / f"mapping_line_{i}.png", bbox_inches="tight")

    # visualize distribution of architectures (best mapping)
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="min",
            target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
            archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    matching_keys = prob_keys
    for i in range(num_layers):
        matching_values = df.loc[i, matching_keys]
        idxs_per_key = [df[matching_keys[i]] == matching_values[i] for i in range(len(matching_values))]
        idxs = idxs_per_key[0]
        for key_idxs in idxs_per_key:
            idxs = np.bitwise_and(idxs, key_idxs)
        comp_rows = df.loc[idxs]
        vals = comp_rows[target_key]
        vals = sorted(vals)[:int(len(vals)*best_part)]
        print(len(vals))
        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        for fig in range(2):
            plt.figure()
            plt.title(prob)
            if fig == 0:
                plt.ylabel("count")
                plt.xlabel(target_key)
                plt.hist(vals, bins=100)
                plt.savefig(output_dir / f"arch_hist_{i}.png", bbox_inches="tight")
            if fig == 1:
                plt.xlabel("arch")
                plt.ylabel(target_key)
                plt.plot(range(len(vals)), list(reversed(sorted(vals))))
                plt.savefig(output_dir / f"arch_line_{i}.png", bbox_inches="tight")

def vis_pareto(dataset_path: pathlib.Path, output_dir: pathlib.Path, num_layers: int, mappings_per_layer: int, 
               target_key_1: str, target_key_2: str, best_part: float = 1.0):
    # first visualize mappings for one arch
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
            target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
            archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = arch_keys + prob_keys
    first_idxs = df.groupby(by=matching_keys, sort=False)["index"].first()
    for i, row_idx in enumerate(first_idxs):
        matching_values = df.loc[row_idx, matching_keys]
        comp_rows = get_matching_rows(df, matching_keys, matching_values)
        vals_1 = comp_rows[target_key_1][:mappings_per_layer]
        vals_2 = comp_rows[target_key_2][:mappings_per_layer]
        best_part = 0.1
        # best_idxs = np.argpartition(vals_1, int(len(vals_1)*best_part)-1)[:int(len(vals_1)*best_part)]
        # best_latencies = vals_1 <= (vals_1.max() / 10)
        # vals_1 = np.array(vals_1)[best_latencies]
        # vals_2 = np.array(vals_2)[best_latencies]
        best_idxs = (vals_1*vals_2) <= ((vals_1*vals_2).min()*10)
        vals_1 = np.array(vals_1)[best_idxs]
        vals_2 = np.array(vals_2)[best_idxs]
        print(len(vals_1))
        if 1 < len(vals_1) < 10:
            print(matching_values)
            print(np.array(comp_rows["mapping.mapping"])[best_idxs])
            print(vals_1)
            print(vals_2)
        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        plt.figure()
        plt.title(prob)
        plt.xlabel(target_key_1)
        plt.ylabel(target_key_2)
        plt.scatter(vals_1, vals_2, s=10)
        min_edp = (vals_1*vals_2).min()
        xlim = plt.gca().get_xlim()
        for iso_edp in [min_edp, min_edp*2, min_edp*3]:
            x = np.linspace(max(iso_edp / vals_2.max(), xlim[0]), xlim[1], 1000)
            y = iso_edp / x
            plt.plot(x, y, label=iso_edp)
        plt.legend()
        plt.savefig(output_dir / f"pareto_{prob}.png", bbox_inches="tight")
        plt.close()

def compare_arch_accuracy(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str, num_layers: int, mappings_per_layer: int):
    """
    Arch search with a small number of mappings per layer
    """
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
        target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
        archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = prob_keys
    for i in range(num_layers): # for each layer
        row_idx = mappings_per_layer*i
        # get rows with same prob as row row_idx
        matching_values = df.loc[row_idx, matching_keys]
        comp_df = get_matching_rows(df, matching_keys, matching_values)
        # now, try comparing arch with just 10 random mappings
        arch_vals_groups = dict(iter(comp_df.groupby(by=arch_keys, sort=False)))
        arch_vals_group_names = list(arch_vals_groups)
        num_mappings_comp = [100, 200, 400, 600, 800, 1000]
        # counts = {num_mappings: {"correct": 0, "total": 0} for num_mappings in num_mappings_comp}
        # for arch1_name_idx in range(len(arch_vals_group_names)):
        #     for arch2_name_idx in range(arch1_name_idx+1, len(arch_vals_group_names)):
        #         arch1_name = arch_vals_group_names[arch1_name_idx]
        #         arch2_name = arch_vals_group_names[arch2_name_idx]
        #         arch1_vals = arch_vals_groups[arch1_name]
        #         arch2_vals = arch_vals_groups[arch2_name]
        #         arch1_target_vals_all_min = arch1_vals[target_key].min()
        #         arch2_target_vals_all_min = arch2_vals[target_key].min()
        #         for num_mappings in num_mappings_comp:
        #             arch1_target_vals_first_k_min = arch1_vals[:num_mappings][target_key].min()
        #             arch2_target_vals_first_k_min = arch2_vals[:num_mappings][target_key].min()
        #             correct = (arch1_target_vals_first_k_min <= arch2_target_vals_first_k_min) == (arch1_target_vals_all_min <= arch2_target_vals_all_min)
        #             counts[num_mappings]["correct"] += int(correct)
        #             counts[num_mappings]["total"] += 1
        # print(counts)
        # comp_accuracies = {n: counts[n]["correct"] / counts[n]["total"] for n in counts}

        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])

        # plt.figure()
        # plt.title(prob)
        # plt.ylabel("arch comparison accuracy")
        # plt.xlabel("mappings compared")
        # plt.plot(comp_accuracies.keys(), comp_accuracies.values())
        # plt.savefig(output_dir / f"arch_comp_{i}.png", bbox_inches="tight")

        # scores = {num_mappings: {"min_first_k": float("inf"), "min_total": float("inf")} for num_mappings in num_mappings_comp}
        # for arch_name in arch_vals_group_names:
        #     arch_vals = arch_vals_groups[arch_name]
        #     arch_target_vals_all_min = arch_vals[target_key][:mappings_per_layer].min()
        #     for num_mappings in num_mappings_comp:
        #         # pick in random order to prevent using the same mappings for each arch,
        #         # since timeloop random mapper uses the same mappings
        #         idxs = random.sample(range(len(arch_vals)), min(num_mappings, len(arch_vals)))
        #         arch_target_vals_first_k_min = arch_vals.iloc[idxs][target_key].min()
        #         if arch_target_vals_first_k_min < scores[num_mappings]["min_first_k"]:
        #             scores[num_mappings]["min_first_k"] = arch_target_vals_first_k_min
        #             scores[num_mappings]["min_total"] = arch_target_vals_all_min
        # plt.figure()
        # plt.title(prob)
        # plt.ylabel(f"best arch min {target_key} (1000 mappings)")
        # plt.xlabel("num mappings used")
        # plt.plot(scores.keys(), [scores[n]["min_total"] for n in scores])
        # plt.savefig(output_dir / f"arch_comp_best_5_{i}.png", bbox_inches="tight")

        # best_after_n_maps = {num_mappings: list() for num_mappings in num_mappings_comp}
        # for arch_name in arch_vals_group_names:
        #     arch_vals = arch_vals_groups[arch_name]
        #     best_so_far = search_curve(arch_vals[target_key].values)
        #     arch_target_vals_all_min = arch_vals[target_key][:mappings_per_layer].min()
        #     for num_mappings in num_mappings_comp:
        #         first_k_min = best_so_far[min(num_mappings, len(best_so_far)-1)]
        #         best_after_n_maps[num_mappings].append(first_k_min)

        # k_best_arch = 5
        # to_plot = dict()
        # for n in best_after_n_maps:
        #     idxs = np.argpartition(best_after_n_maps[n], k_best_arch)[:k_best_arch]
        #     target_achieved = np.array(best_after_n_maps[mappings_per_layer])[idxs].min()
        #     to_plot[n] = target_achieved

        # plt.figure()
        # plt.title(prob)
        # plt.ylabel(f"best arch min {target_key} (1000 mappings total, {k_best_arch} arch searched)")
        # plt.xlabel("num mappings used")
        # plt.plot(to_plot.keys(), to_plot.values())
        # plt.savefig(output_dir / f"arch_comp_best_{k_best_arch}_{i}.png", bbox_inches="tight")

def random_mapping_trajectories(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str, num_layers: int, mappings_per_layer: int):
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
        target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
        archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = prob_keys
    for i in range(num_layers): # for each layer
        row_idx = mappings_per_layer*i
        # get rows with same prob as row row_idx
        matching_values = df.loc[row_idx, matching_keys]
        comp_df = get_matching_rows(df, matching_keys, matching_values)
        arch_vals_groups = dict(iter(comp_df.groupby(by=arch_keys, sort=False)))
        arch_vals_group_names = list(arch_vals_groups)
        
        tries = 0
        pairs_tried = set()
        pairs_comp = set()
        while len(pairs_comp) < 5:
            if tries > len(arch_vals_group_names) ** 2:
                break
            tries += 1
            arch1_idx, arch2_idx = random.sample(range(len(arch_vals_group_names)), 2)
            if (arch1_idx, arch2_idx) in pairs_tried:
                continue
            pairs_tried.add((arch1_idx, arch2_idx))
            arch1_name = arch_vals_group_names[arch1_idx]
            arch2_name = arch_vals_group_names[arch2_idx]
            arch1_vals = arch_vals_groups[arch1_name][target_key][:mappings_per_layer]
            arch2_vals = arch_vals_groups[arch2_name][target_key][:mappings_per_layer]
            ratio = arch1_vals.min() / arch2_vals.min()
            if ratio < 0.5 or ratio > 2:
                continue
            pairs_comp.add((arch1_idx, arch2_idx))

            prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])

            plt.figure()
            plt.title(prob)
            plt.ylabel(target_key)
            plt.xlabel("random mappings searched")
            arch1_vals_shuffled = random.sample(list(arch1_vals.values), len(arch1_vals))
            arch2_vals_shuffled = random.sample(list(arch2_vals.values), len(arch2_vals))
            plt.plot(range(len(arch1_vals))[100:], search_curve(arch1_vals_shuffled)[100:], label=str(arch1_name))
            plt.plot(range(len(arch2_vals))[100:], search_curve(arch2_vals_shuffled)[100:], label=str(arch2_name))
            plt.legend()
            plt.savefig(output_dir / f"random_traj_layer{i}_arch{arch1_idx}_arch{arch2_idx}.png", bbox_inches="tight")

def arch_search_function(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str, num_layers: int, mappings_per_layer: int, 
                         mapping_agg_fn: Callable[[Iterable], float], fn_name: str):
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
        target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
        archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = prob_keys
    for i in range(num_layers): # for each layer
        row_idx = mappings_per_layer*i
        # get rows with same prob as row row_idx
        matching_values = df.loc[row_idx, matching_keys]
        comp_df = get_matching_rows(df, matching_keys, matching_values)
        # now, try comparing arch with diff numbers of mappings
        arch_vals_groups = dict(iter(comp_df.groupby(by=arch_keys, sort=False)))
        arch_vals_group_names = list(arch_vals_groups)
        num_mappings_comp = [100, 200, 400, 600, 800, 1000]

        agg_vals_after_n_maps = {num_mappings: list() for num_mappings in num_mappings_comp}
        for arch_name in arch_vals_group_names:
            arch_vals = arch_vals_groups[arch_name]
            for num_mappings in num_mappings_comp:
                agg_val_so_far = mapping_agg_fn(arch_vals[target_key][:num_mappings])
                agg_vals_after_n_maps[num_mappings].append(agg_val_so_far)

        k_best_arch = 5
        to_plot = dict()
        mins = []
        for group in arch_vals_groups.values():
            mins.append(group[target_key].min())
        mins = np.array(mins)
        for n in agg_vals_after_n_maps:
            idxs = np.argpartition(agg_vals_after_n_maps[n], k_best_arch)[:k_best_arch]
            target_achieved = mins[idxs].min()
            to_plot[n] = target_achieved
        best_known_idx = mins.argmin()
        best_known_val = mins[best_known_idx]
        name = arch_vals_group_names[best_known_idx]
        best_known_dist = list(reversed(sorted(arch_vals_groups[name][target_key][:mappings_per_layer].values)))

        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        plt.figure()
        plt.title(prob)
        plt.ylabel(f"{target_key}")
        plt.xlabel("mapping #")
        plt.plot(range(len(best_known_dist[100:])), best_known_dist[100:], "-", color="orange", label=str(arch_vals_group_names[best_known_idx]))
        idxs = np.argpartition(agg_vals_after_n_maps[1000], k_best_arch)[:k_best_arch]
        for idx in idxs:
            name = arch_vals_group_names[idx]
            dist = list(reversed(sorted(arch_vals_groups[name][target_key][:mappings_per_layer].values)))
            plt.plot(range(len(dist[100:])), dist[100:], "b:", label=str(name))
        plt.legend()
        plt.savefig(output_dir / f"arch_comp_{fn_name}_best_{k_best_arch}_{i}_legend.png", bbox_inches="tight")

        plt.figure()
        plt.title(prob)
        plt.ylabel(f"best arch min {target_key} ({mappings_per_layer} mappings total, {k_best_arch} arch searched)")
        plt.xlabel("num mappings used")
        plt.plot(to_plot.keys(), to_plot.values())
        plt.plot(to_plot.keys(), [best_known_val] * len(to_plot))
        plt.savefig(output_dir / f"arch_comp_{fn_name}_best_{k_best_arch}_{i}_no_curve.png", bbox_inches="tight")


def arch_search_function_network(dataset_path: pathlib.Path, output_dir: pathlib.Path, target_key: str, num_layers: int, mappings_per_layer: int, 
                         mapping_agg_fn: Callable[[Iterable], float], fn_name: str, layer_count: pathlib.Path):
    dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=False, total_samples=0, split_ratios={"train": 1}, process_mappings="",
        target_log=False, target_norm=None, probfeat_log=False, probfeat_norm=None,
        archfeat_log=False, archfeat_norm=None, mapfeat_log=False, mapfeat_norm=None)
    df = dla_dataset_creator.train_data.df
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = prob_keys
    for i in range(num_layers): # for each layer
        row_idx = mappings_per_layer*i
        # get rows with same prob as row row_idx
        matching_values = df.loc[row_idx, matching_keys]
        comp_df = get_matching_rows(df, matching_keys, matching_values)
        # now, try comparing arch with diff numbers of mappings
        arch_vals_groups = dict(iter(comp_df.groupby(by=arch_keys, sort=False)))
        arch_vals_group_names = list(arch_vals_groups)
        num_mappings_comp = [100, 200, 400, 600, 800, 1000]

        agg_vals_after_n_maps = {num_mappings: list() for num_mappings in num_mappings_comp}
        for arch_name in arch_vals_group_names:
            arch_vals = arch_vals_groups[arch_name]
            for num_mappings in num_mappings_comp:
                agg_val_so_far = mapping_agg_fn(arch_vals[target_key][:num_mappings])
                agg_vals_after_n_maps[num_mappings].append(agg_val_so_far)

        k_best_arch = 5
        to_plot = dict()
        mins = []
        for group in arch_vals_groups.values():
            mins.append(group[target_key].min())
        mins = np.array(mins)
        for n in agg_vals_after_n_maps:
            idxs = np.argpartition(agg_vals_after_n_maps[n], k_best_arch)[:k_best_arch]
            target_achieved = mins[idxs].min()
            to_plot[n] = target_achieved
        best_known_idx = mins.argmin()
        best_known_val = mins[best_known_idx]
        name = arch_vals_group_names[best_known_idx]
        best_known_dist = list(reversed(sorted(arch_vals_groups[name][target_key][:mappings_per_layer].values)))

        prob = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        plt.figure()
        plt.title(prob)
        plt.ylabel(f"{target_key}")
        plt.xlabel("mapping #")
        plt.plot(range(len(best_known_dist[100:])), best_known_dist[100:], "-", color="orange", label=str(arch_vals_group_names[best_known_idx]))
        idxs = np.argpartition(agg_vals_after_n_maps[1000], k_best_arch)[:k_best_arch]
        for idx in idxs:
            name = arch_vals_group_names[idx]
            dist = list(reversed(sorted(arch_vals_groups[name][target_key][:mappings_per_layer].values)))
            plt.plot(range(len(dist[100:])), dist[100:], "b:", label=str(name))
        plt.legend()
        plt.savefig(output_dir / f"arch_comp_{fn_name}_best_{k_best_arch}_{i}_legend.png", bbox_inches="tight")

        plt.figure()
        plt.title(prob)
        plt.ylabel(f"best arch min {target_key} ({mappings_per_layer} mappings total, {k_best_arch} arch searched)")
        plt.xlabel("num mappings used")
        plt.plot(to_plot.keys(), to_plot.values())
        plt.plot(to_plot.keys(), [best_known_val] * len(to_plot))
        plt.savefig(output_dir / f"arch_comp_{fn_name}_best_{k_best_arch}_{i}_no_curve.png")

if __name__ == "__main__":
    start_time = time.time()
    try:
        utils.set_random_seed(0)
        # dataset_path = _DATA_DIR / "output_random_gemmini_mm_conv_7_22_22" / "dataset.csv"
        # dataset_path = _DATA_DIR / "output_random_simba_mm_conv_8_1_22" / "dataset.csv"
        # dataset_path = _DATA_DIR / "gemmini_resnet50_1000arch_1000map_11_1_22" / "dataset.csv"
        # dataset_path = _DATA_DIR / "gemmini_resnet50_defaultarch_1000map_11_14_22" / "dataset.csv"
        dataset_path = _DATA_DIR / "gemmini_resnet50_defaultarch_10000map_12_6_22" / "dataset.csv"
        dataset_path = _DATA_DIR / "gemmini_resnet50_biggerarch_1000map_3_7_23" / "dataset.csv"
        # cosa_dataset_path = _DATA_DIR / "gemmini_resnet50_defaultarch_cosamap_12_6_22" / "dataset.csv"
        # dataset_path = _DATA_DIR / "gemmini_resnet50_biggerarch_1000map_3_3_23" / "dataset.csv"
        # dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=True, total_samples=2_000_000, split_ratios={"train": 1}, process_mappings="split",
        #         target_log=True, target_norm="mean", probfeat_log=False, probfeat_norm="mean",
        #         archfeat_log=True, archfeat_norm="mean", mapfeat_log=False, mapfeat_norm="mean", num_processes=16)
        # df: pd.DataFrame = dla_dataset_creator.train_data.df
        # mapping_keys = utils.keys_by_type(df, "mapping")
        # norm_arch = dla_dataset_creator.train_data.norm("arch", torch.tensor([16, 256, 64], dtype=torch.float32, requires_grad=True))
        # print(norm_arch)
        # denorm_arch = dla_dataset_creator.train_data.denorm("arch", norm_arch)
        # print(denorm_arch)
        # print(denorm_arch.requires_grad)
        output_dir = pathlib.Path("vis_pareto")
        target_key = "target.cycle"
        num_layers = 23
        mappings_per_layer = 10000
        output_dir.mkdir(exist_ok=True)
        # pred_perf(dataset_path, output_dir, target_key)
        # vis_mappings(dataset_path, output_dir, target_key, num_layers, mappings_per_layer, best_part=0.95, cosa_dataset_path=cosa_dataset_path, use_search_curve=True)
        vis_pareto(dataset_path, output_dir, num_layers, mappings_per_layer, "target.cycle", "target.energy")
        # compare_arch_accuracy(dataset_path, output_dir, target_key, num_layers, mappings_per_layer)
        # random_mapping_trajectories(dataset_path, output_dir, target_key, num_layers, mappings_per_layer)
        # def mapping_agg_fn(lst):
        #     num_to_mean = len(lst) // 10
        #     idxs = np.argpartition(lst, num_to_mean)[:num_to_mean]
        #     return np.array(lst)[idxs].mean() 
        # fn_name = "top10pmean"
        # arch_search_function(dataset_path, output_dir, target_key, num_layers, mappings_per_layer, mapping_agg_fn, fn_name)
        # arch_search_function_network(dataset_path, output_dir, target_key, num_layers, mappings_per_layer, mapping_agg_fn, fn_name,
        #                      layer_count=DATASET_ROOT_PATH/"workloads"/"resnet50"/"layer_count.yaml")

        # data_creator = DlaDatasetCreator(dataset_path=dataset_path, 
        #                                  split_ratios={"train": 1},
        #                                  shuffle=False,
        #                                  total_samples=1, 
        #                                  process_mappings="arr")
        # train_data = data_creator.train_data
        # logger.info("Training data took %.2f seconds to load", time.time() - start_time)

        # output_dir = pathlib.Path(__file__).parent.parent.parent.resolve() / "output_dir_eval_test"
        # layer_path = pathlib.Path(__file__).parent.parent.resolve() / "workloads" / "conv" / "conv_0.yaml"
        # prob = Prob(layer_path)
        # print(train_data.df)
        # example_mapping = train_data.df.loc[0, "mapping.flat_mapping"]
        # eval_result = eval.eval_layer(output_dir, "gemmini", [8, 20, 10], prob, mapping=example_mapping)
        # logger.info("Evaluation result: %s", eval_result)

        # import pandas as pd
        # pd.set_option('display.max_columns', None)
        # logger.debug(train_data.df.head(3))
        # logger.debug(train_data[10])
    except Exception:
        exc = traceback.format_exc()
        logger.error(exc)
    # pred_perf(dataset_path)
