import pathlib
import traceback
import math
import random
import multiprocessing
import time
from functools import lru_cache
import logging

import skopt
from skopt.plots import plot_convergence
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
import swifter
# import wandb
import sklearn
import scipy

from dataset import DATASET_ROOT_PATH
from dataset.common import logger, utils, mapping_utils
from dataset.hw import init_hw_config, HardwareConfig, GemminiConfig
from dataset.workloads import Prob
from dataset.dse import predictors, pytorch_util, DlaDatasetCreator, eval, energy_model, latency_model, bo

@lru_cache(None)
def log_once(msg: str):
    logger.debug(msg)

class TrivialPointGenerator(skopt.sampler.InitialPointGenerator):
    def __init__(self, points):
        self.points = points
    
    def generate(self, dimensions, n_samples, random_state=None):
        samples = []
        for i in range(n_samples):
            samples.append(self.points[i % len(self.points)])
        return samples


class CosaPointGenerator(skopt.sampler.InitialPointGenerator):
    """
    An skopt InitialPointGenerator that generates a random HW config, then uses CoSA to generate
    a mapping for a given layer and that HW config.
    """
    saved_arch_configs = []

    def __init__(self, arch_name: str, output_dir: pathlib.Path, prob: Prob, save_hw: bool = False):
        """
        save_hw option determines whether the same HW is used each time the generator
        is instantiated.
        """
        self.arch_name = arch_name
        self.prob = prob
        self.output_dir = output_dir
        self.save_hw = save_hw # use saved HW and save any new generated ones

    @classmethod
    def save_arch_config(cls, arch_config: HardwareConfig):
        cls.saved_arch_configs.append(arch_config)

    @classmethod
    def get_saved_arch_configs(cls):
        return cls.saved_arch_configs

    @classmethod
    def reset_saved_arch_configs(cls):
        cls.saved_arch_configs = list()
    
    def generate(self, dimensions, n_samples, random_state=None):
        samples = []
        for i in range(n_samples):
            saved_arch_configs = CosaPointGenerator.get_saved_arch_configs()
            if self.save_hw and i < len(saved_arch_configs):
                arch_config = saved_arch_configs[i]
            else:
                arch_config = init_hw_config(self.arch_name, "random", self.output_dir)
            # flat_mapping = arch_config.run_cosa(self.prob, run_mapping=False)

            row = arch_config.run_cosa(self.prob, run_mapping=False)[0]
            flat_mapping = list(mapping_utils.process_mapping(row["mapping.mapping"], self.prob.shape))
            # logger.info(flat_mapping)
            # logger.info(flat_mapping_with_run)
            # assert (list(flat_mapping) == flat_mapping_with_run)
            samples.append(flat_mapping)
            if self.save_hw:
                CosaPointGenerator.save_arch_config(arch_config)

        return samples

    def generate_eval(self, dimensions, n_samples, random_state=None, hw_config="random"):
        samples = []
        for i in range(n_samples):
            saved_arch_configs = CosaPointGenerator.get_saved_arch_configs()
            if self.save_hw and i < len(saved_arch_configs):
                arch_config = saved_arch_configs[i]
            else:
                arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
            # flat_mapping = arch_config.run_cosa(self.prob, run_mapping=False)

            row = arch_config.run_cosa(self.prob, run_mapping=True)[0]
            energy = row["target.energy"]
            cycle = row["target.cycle"]
            logger.debug("Energy: %s, Cycle: %s", energy, cycle)
            flat_mapping = list(mapping_utils.process_mapping(row["mapping.mapping"], self.prob.shape))
            # logger.info(flat_mapping)
            # logger.info(flat_mapping_with_run)
            # assert (list(flat_mapping) == flat_mapping_with_run)
            samples.append(flat_mapping)
            if self.save_hw:
                CosaPointGenerator.save_arch_config(arch_config)

        return samples, energy, cycle


class MappingDrivenNetworkSearcher():
    def __init__(self, arch_name: str, output_dir: pathlib.Path, workload: str, metric: str, gpu_id=0, log_times=False):
        logger.info("Initializing Network Searcher with params %s", locals())
        self.arch_name = arch_name
        self.output_dir = pathlib.Path(output_dir).resolve()
        print(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layers: list[Prob] = []
        base_workload_path = DATASET_ROOT_PATH / "workloads"
        self.workload_path = base_workload_path / workload
        self.workload_name = workload
        self.metric = metric
        self.log_times = log_times

        unique_layers = utils.parse_yaml(self.workload_path / 'unique_layers.yaml')
        for unique_layer in unique_layers:
            layer_path = self.workload_path / (unique_layer+'.yaml')
            layer_path = layer_path.resolve()
            self.layers.append(Prob(layer_path))

        self.layer_count = self.get_layer_count()
        logger.info("Layer count: %s", self.layer_count)

        num_dims = 7
        prob_dims = [[] for _ in range(num_dims)] # one entry per dim, nested lists contain one entry per layer
        for prob in self.layers:
            for dim_idx, dim in prob.prob_idx_name_dict.items():
                prob_dims[dim_idx].append(prob.prob[dim])

        self.gpu_id = gpu_id
        pytorch_util.init_gpu(gpu_id=self.gpu_id)
        self.prob_dims_recip = None
        for dim_idx in range(len(prob_dims)):
            one_dim = pytorch_util.from_numpy(1/np.array(prob_dims[dim_idx])).unsqueeze(0)
            if self.prob_dims_recip is None:
                self.prob_dims_recip = one_dim
            else:
                self.prob_dims_recip = torch.cat((self.prob_dims_recip, one_dim), dim=0)

    def get_layer_count(self) -> list[int]:
        try:
            layer_count_dict = utils.parse_yaml(self.workload_path / 'layer_count.yaml')
            counts = [layer_count_dict[prob.config_str()]["count"] for prob in self.layers]
        except:
            logger.warning("Couldn't find layer count, using default layer counts")
            counts = [1 for prob in self.layers]
        return counts

    def search_bo(self, n_calls: int, n_initial_points: int, iters_per_layer: int = 10):
        """
        Use Bayesian optimization to search mappings for all layers
        """
        # Initialize mappings
        mappings = []
        results = []
        for prob in self.layers:
            cosa_gen = CosaPointGenerator(self.arch_name, self.output_dir, prob, save_hw=True)
            mapping_in_lst = cosa_gen.generate(None, 1)
            mappings.extend(mapping_in_lst)

        def select_idx_fn(cur_prob_idx):
            cur_prob_idx = cur_prob_idx[0]
            prob = self.layers[cur_prob_idx]
            def minimize_mapping_fn(mapping):
                mapping_prob_pairs = [(mappings[i], self.layers[i],) for i in range(len(self.layers))]
                mapping = mapping_utils.round_mapping(mapping, prob)
                mapping_prob_pairs[cur_prob_idx] = (mapping, prob,)

                hw_config, cap_per_layer, max_cap_idxs = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
                arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
                row = arch_config.run_mapping_from_dict(prob, arch_config.flat_mapping_to_dict(prob.shape, mapping))
                try:
                    target = row["target.cycle"] * row["target.energy"]
                except Exception:
                    logger.error("Could not run min_hw for rounded mapping %s on arch %s, prob %s\n%s",
                                    mapping, self.arch_name, prob.config_str(), traceback.format_exc())
                    target = np.finfo("float32").max
                return target

            bounds = get_mapping_bounds(self.arch_name, prob)
            logger.debug("BO search bounds: %s", bounds)
            gen = TrivialPointGenerator([mappings[cur_prob_idx]])
            initial_point_generator = skopt.utils.cook_initial_point_generator(gen)
            res_gp = skopt.gp_minimize(minimize_mapping_fn,
                                    bounds, 
                                    n_calls=n_calls,
                                    n_initial_points=n_initial_points,
                                    n_jobs=16,
                                    initial_point_generator=initial_point_generator,
                                    random_state=0)
            mappings[cur_prob_idx] = res_gp.x # update mapping to new best
            results.append(res_gp.fun)
            return res_gp.fun

        idx_bounds = [skopt.space.Categorical(range(len(self.layers)), name="cur_prob_idx")]
        res_gp = skopt.gp_minimize(select_idx_fn,
                                idx_bounds,
                                n_calls=iters_per_layer, # Optimize for a set number of iterations
                                n_initial_points=len(self.layers),
                                n_jobs=16,
                                random_state=0)
        return mappings, res_gp.func_vals

    def search_bo_v2(self, n_calls: int, n_initial_points: int, iters_per_layer: int = 10):
        # Initialize mappings
        mappings = []
        results = []
        for prob in self.layers:
            cosa_gen = CosaPointGenerator(self.arch_name, self.output_dir, prob, save_hw=True)
            mapping_in_lst = cosa_gen.generate(None, 1)
            mappings.extend(mapping_in_lst)

        mapping_prob_pairs = [(mappings[i], self.layers[i],) for i in range(len(self.layers))]
        hw_config, cap_per_layer, max_cap_idxs = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
        for it in range(iters_per_layer*len(self.layers)):
            cur_prob_idx = self.select_prob_fn(mappings, hw_config, cap_per_layer, max_cap_idxs)
            logger.info("Iteration %s, now optimizing layer %s", it, cur_prob_idx)
            prob = self.layers[cur_prob_idx]
            def minimize_mapping_fn(mapping):
                mapping_prob_pairs = [(mappings[i], self.layers[i],) for i in range(len(self.layers))]
                mapping = mapping_utils.round_mapping(mapping, prob)
                mapping_prob_pairs[cur_prob_idx] = (mapping, prob,)

                nonlocal hw_config, cap_per_layer, max_cap_idxs
                hw_config, cap_per_layer, max_cap_idxs = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
                arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
                cycle = 0
                area = 0
                for mapping_i, prob_i in mapping_prob_pairs:
                    row = arch_config.run_mapping_from_dict(prob_i, arch_config.flat_mapping_to_dict(prob_i.shape, mapping_i))
                    try:
                        # target = row["target.cycle"] * row["target.energy"]
                        cycle += row["target.cycle"]
                        area = row["target.area"]
                    except Exception:
                        logger.error("Could not run min_hw for rounded mapping %s on arch %s, prob %s\n%s",
                                        mapping, self.arch_name, prob.config_str(), traceback.format_exc())
                        # target = np.finfo("float32").max
                        cycle += np.finfo("float32").max
                return cycle * area

            bounds = get_mapping_bounds(self.arch_name, prob)
            logger.debug("BO search bounds: %s", bounds)
            gen = TrivialPointGenerator([mappings[cur_prob_idx]])
            initial_point_generator = skopt.utils.cook_initial_point_generator(gen)
            res_gp = skopt.gp_minimize(minimize_mapping_fn,
                                    bounds, 
                                    n_calls=n_calls,
                                    n_initial_points=n_initial_points,
                                    n_jobs=16,
                                    initial_point_generator=initial_point_generator,
                                    random_state=0)
            mappings[cur_prob_idx] = res_gp.x # update mapping to new best
            results.append(res_gp.fun)

        mapping_prob_pairs = [(mappings[i], self.layers[i],) for i in range(len(self.layers))]
        hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
        return hw_config, mappings, results

    # def search_random_input(self, dataset_creator, random_mappings):
    #     gpu_id=0
    #     pytorch_util.init_gpu(gpu_id=gpu_id)
    #     big_arch = GemminiConfig([2048, 2**20, 2**20], self.output_dir)
    #     for prob in self.layers:
    #         rows = big_arch.run_random_mappings(prob, 10000)
    #         for row in rows:

    def search_gd(self, dataset_creator, random_arch_configs, **kwargs):
        num_starts = kwargs["num_starts"]
        sgd_lr = kwargs["sgd_lr"]
        base_gd_iters = kwargs["base_gd_iters"]
        per_start_gd_step = kwargs["per_start_gd_step"]
        round_steps = kwargs["round_steps"]
        ordering_search_method = kwargs["ordering_search_method"]
        gumbel_temp = kwargs["gumbel_temp"]
        shuffle_perms = kwargs["shuffle_perms"]
        extra_test_data = kwargs["extra_test_data"]
        latency_predictor_type = kwargs["predictor"]
        plot_only = kwargs["plot_only"]
        constant_pe = kwargs["constant_pe"]
        
        train_dataset = dataset_creator.get_train_data()
        test_dataset = dataset_creator.get_test_data()
        # min_hw_1 = train_dataset.df["dse.min_hw_1"]
        # logger.debug("min_hw_1 mean: %s, std: %s", min_hw_1.mean(), min_hw_1.std())

        # mapping_str = "L3[WIO] Q7 K8 - L2[WI] N1 K4X - L1[O] K2 C4 P56 Q8 C16X - L0[W] N1"
        # mapping_str = "L3[WIO] C3 K8 Q4 P16 - L2[WI] P7 K2X - L1[O] K4 Q7 S7 R7 - L0[W] Q4"
        # mapping_str = "L3[WIO] K16 C2 S3 R3 - L2[WI] N1 K8X - L1[O] C32 Q7 K4 C8X - L0[W] P7"
        # mapping_str = "L3[WIO] K125 - L2[WI] N1 K8X - L1[O] C256 C8X - L0[W] N1"
        # mapping_str = "L3[WIO] Q5 K32 P11 R11 S11 - L2[WI] C3 K2X - L1[O] P5 Q11 - L0[W] N1"
        # mapping_str = "L3[WIO] R2 K256 C4 Q7 S2 - L2[WI] N1 - L1[O] C32 Q2 P14 C2X - L0[W] N1" # inner["S"] - 1 is correct
        # mapping_str = "L3[WIO] K32 P12 R3 C3 Q36 S3 - L2[WI] N1 - L1[O] K2 Q3 P3 - L0[W] P3" # inner["S"] is correct
        mapping_str = "L3[WIO] S3 C16 R3 K64 - L2[WI] N1 K4X - L1[O] K2 Q14 P14 - L0[W] N1"
        # mapping_str = "L3[WIO] C2 - L2[WI] N1 - L1[O] C16 C8X - L0[W] N1"
        flat_mapping = pytorch_util.from_numpy(mapping_utils.process_mapping(mapping_str, "cnn-layer"))
        example_flat_mapping_normed = train_dataset.norm("mapping", flat_mapping)
        logger.info("denormed mapping: %s", flat_mapping)
        logger.info("normed mapping: %s", example_flat_mapping_normed)
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv_test" / "conv_0.yaml")
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "resnext50_32x4d" / "_outputs_input.2.yaml")
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "alexnet" / "_outputs_input.8.yaml")
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "mm" / "mm_0.yaml") # 1024 x 1024 x 1024
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "resnet50" / "_outputs_input.8.yaml")
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "bert" / "mm_0.yaml")
        # prob = Prob(DATASET_ROOT_PATH / "workloads" / "dlrm" / "_outputs_213.yaml")
        prob = Prob({
            "problem": {
                "C": 16,
                "Hdilation": 1,
                "Hstride": 2,
                "K": 64 * 4 * 2,
                "N": 1,
                "P": 14,
                "Q": 14,
                "R": 3,
                "S": 3,
                "Wdilation": 1,
                "Wstride": 2,
            },
            "shape": "cnn-layer"})
        # mapping_min_hw, _, _ = min_hw(self.arch_name, self.output_dir, [(flat_mapping, prob,)], grad=True)
        # flat_mapping.requires_grad = True

        # arch_config = GemminiConfig([2048, 2**20, 2**20], self.output_dir)
        # arch_config = GemminiConfig([128, 1, 1], self.output_dir)
        # mapping_dict = arch_config.flat_mapping_to_dict("cnn-layer", mapping_utils.process_mapping(mapping_str, "cnn-layer"))
        # arch_config.run_mapping_from_dict(prob, mapping_dict)

        # reads, updates, writes = mapping_utils.accesses_from_mapping(flat_mapping, prob)
        # _, _, capacities = mapping_utils.capacity_from_mapping(flat_mapping, prob)
        # print("capacities", capacities)
        # print("reads", reads)
        # print("updates", updates)
        # print("writes", writes)
        # exit(0)

        # Tianrui is the GOAT

        ##### Get some relevant normalization stats
        stats = train_dataset.creator.stats
        latency_max = stats["target.cycle_max"]
        energy_max = stats["target.energy_max"]
        # latency_max = 1
        # energy_max = 1
        # latency_min = stats["target.cycle_min"]
        # latency_mean = stats["target.cycle_mean"]
        # latency_std = stats["target.cycle_std"]
        # area_mean = stats["target.area_mean"]
        # area_std = stats["target.area_std"]
        # energy_mean = stats["target.energy_mean"]
        # energy_std = stats["target.energy_std"]

        ##### Find which mapping parameters actually change
        mapping_keys = utils.keys_by_type(train_dataset.df, "mapping")
        spatial_idxs = []
        temporal_idxs = []
        for idx in range(len(mapping_keys)):
            if "spatial" in mapping_keys[idx]:
                spatial_idxs.append(idx)
            if "temporal" in mapping_keys[idx]:
                temporal_idxs.append(idx)
        min_hw_net_mapping_idxs = [
            spatial_idxs,
            temporal_idxs + spatial_idxs,
            temporal_idxs + spatial_idxs,
        ]

        relevant_idxs = []
        relevant_keys = []
        relevant_keys = mapping_keys
        relevant_idxs = list(range(len(mapping_keys)))
        # for idx, key in enumerate(mapping_keys):
        #     if stats[key+"_std"] != 0:
        #         relevant_idxs.append(idx)
        #         relevant_keys.append(key)
        min_hw_net_mapping_idxs = [
            set(spatial_idxs) & set(relevant_idxs),
            set(temporal_idxs + spatial_idxs) & set(relevant_idxs),
            set(temporal_idxs + spatial_idxs) & set(relevant_idxs),
        ]
        logger.debug("relevant_keys: %s", relevant_keys)
        min_hw_net_mapping_keys = []
        for idx_lst in min_hw_net_mapping_idxs:
            keys = []
            for idx in idx_lst:
                keys.append(mapping_keys[idx])
            min_hw_net_mapping_keys.append(keys)
        min_hw_idx_in_relevant = []
        for min_hw_idx_lst in min_hw_net_mapping_idxs:
            idxs = []
            for i in range(len(relevant_idxs)):
                if relevant_idxs[i] in min_hw_idx_lst:
                    idxs.append(i)
            min_hw_idx_in_relevant.append(idxs)
        
        with_cache = False
        energy_predictor = energy_model.EnergyModel(self.output_dir, 3)
        energy_predictor.train(train_dataset, valid_data=test_dataset, gpu_id=self.gpu_id, num_iters=1000, with_cache=with_cache)
        energy_predictor.train(train_dataset, valid_data=extra_test_data, gpu_id=self.gpu_id, num_iters=1000, with_cache=with_cache)

        access_keys = utils.keys_by_type(train_dataset.df, "dse.access")
        # for key in access_keys:
        #     plt.figure()
        #     plt.scatter(train_dataset.df[key], train_dataset.df["target.cycle"], s=0.2)
        #     plt.xlabel(key)
        #     plt.ylabel("cycles")
        #     plt.title(f"{key} vs latency")
        #     plt.savefig(self.output_dir / utils.unique_filename("png", key), bbox_inches="tight")
        # exit(0)

        with_analytical=True
        with_roofline=False
        if "dnn" in latency_predictor_type:
            with_analytical=False
            with_roofline=False
        train_model=True
        if "dnn" in latency_predictor_type or "both" in latency_predictor_type:
            train_model=True
        # if "noroofline" in latency_predictor_type:
        #     with_roofline=False

        latency_predictor = latency_model.LatencyModel(self.output_dir, relevant_keys)
        latency_predictor.train(train_dataset, valid_data=test_dataset, gpu_id=self.gpu_id, train_model=train_model, with_analytical=with_analytical, with_roofline=with_roofline,
                                continue_training=False, num_iters=10000, interp_points=0, with_cache=with_cache)
        latency_predictor.freeze()
        # latency_golden = latency_model.LatencyModel(self.output_dir, relevant_keys)
        # latency_golden.train(train_dataset, valid_data=test_dataset, gpu_id=self.gpu_id, train_model=False, continue_training=False, with_cache=with_cache)
        # latency_golden.freeze()

        if plot_only:
            # l_test, l_pred, l_losses = latency_predictor.test(train_dataset, gpu_id=self.gpu_id, num_unplot_points=len(train_dataset)//2*0)
            # df = pd.DataFrame.from_dict({"gemmini": l_test.flatten(), "timeloop": l_pred.flatten()})
            # # df.to_csv("/scratch/charleshong/dla-dataset/data/firesim_results_compare.csv")
            # df = pd.DataFrame({"timeloop": l_test.flatten(), "model": l_pred.flatten()})
            # # df = df[df['timeloop'] <= 1e9]
            # df.to_csv(self.output_dir / "latency_train.csv")
            # plt.figure()
            # plt.scatter(df["timeloop"], df["model"], s=0.3)
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.xlabel("real (cycles)")
            # plt.ylabel("pred (cycles)")
            # r2 = sklearn.metrics.r2_score(df["timeloop"], df["model"])
            # plt.title(f"Latency Predictions (Training Set), r^2 = {round(r2,2)}")
            # plt.xlim(1, 10e10)
            # plt.ylim(1, 10e10)
            # plt.savefig(self.output_dir / utils.unique_filename("png", "latency_train"), bbox_inches="tight")
            # # exit(0)
            if "timeloop_dataset" in str(train_dataset.creator.params["dataset_path"]):
                l_test, l_pred, l_losses = latency_predictor.test(test_dataset, gpu_id=self.gpu_id)
                # plt.figure()
                # plt.scatter(l_test, l_pred, s=0.3)
                # plt.xlabel("Timeloop Latency (cycles)")
                # plt.ylabel("Differentiable Model Latency (cycles)")
                # r2 = sklearn.metrics.r2_score(l_test, l_pred)
                # plt.title(f"Latency Model Correlation, r^2 = {round(r2,2)}")
                # plt.savefig(self.output_dir / "correlation_latency.png", bbox_inches="tight")

                e_test, e_pred, e_losses = energy_predictor.test(test_dataset, gpu_id=self.gpu_id)
                # plt.figure()
                # plt.scatter(e_test, e_pred, s=0.3)
                # plt.xlabel("Timeloop Energy (uJ)")
                # plt.ylabel("Differentiable Model Energy (uJ)")
                # r2 = sklearn.metrics.r2_score(e_test, e_pred)
                # plt.title(f"Energy Model Correlation, r^2 = {round(r2,2)}")
                # plt.savefig(self.output_dir / "correlation_energy.png", bbox_inches="tight")

                # plt.figure()
                # plt.scatter(l_test * e_test, l_pred * e_pred, s=0.3)
                # plt.xlabel("Timeloop EDP (uJ * cycles)")
                # plt.ylabel("Differentiable Model EDP (uJ * cycles)")
                # r2 = sklearn.metrics.r2_score(l_test * e_test, l_pred * e_pred)
                # plt.title(f"EDP Correlation, r^2 = {round(r2,2)}")
                # plt.savefig(self.output_dir / "correlation_edp.png", bbox_inches="tight")

                units = {
                    "Latency": "cycles",
                    "Energy": "uJ",
                    "EDP": "uJ * cycles",
                }
                for metric in ["Latency", "Energy", "EDP"]:
                    if metric == "Latency":
                        pred = l_pred
                        test = l_test
                    elif metric == "Energy":
                        pred = e_pred
                        test = e_test
                    elif metric == "EDP":
                        pred = l_pred * e_pred
                        test = l_test * e_test
                    pred = np.array(pred)
                    test = np.array(test)
                    plt.figure()
                    error = (pred - test)/test*100
                    plt.scatter(test, error , s=1)
                    plt.xscale("log")
                    plt.ylim(-15, 15)
                    plt.xlabel(f"Timeloop {metric} ({units[metric]})")
                    plt.ylabel(f"Differentiable Model Error (%)")
                    plt.title(f"{metric} Model Error (MAE = {round(np.absolute(error).mean().item(),2)}%)")
                    plt.grid(axis="y")
                    plt.savefig(self.output_dir / f"error_{metric.lower()}.png", bbox_inches="tight")

                exit(0)

            l_test, l_pred, l_losses = latency_predictor.test(test_dataset, gpu_id=self.gpu_id)
            df = pd.DataFrame({"timeloop": l_test.flatten(), "model": l_pred.flatten()})
            df.to_csv(self.output_dir / "latency_test.csv")
            # df = df[df['timeloop'] <= 1e9]
            plt.figure()
            plt.scatter(df["timeloop"], df["model"], s=0.5)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(10, 10e10)
            plt.ylim(10, 10e10)
            r2 = sklearn.metrics.r2_score(df["timeloop"], df["model"])
            sc = scipy.stats.spearmanr(df["timeloop"], df["model"]).correlation
            plt.xlabel("real (cycles)")
            plt.ylabel("pred (cycles)")
            plt.title(f"Latency Predictions (Test Set), corr.={round(sc,2)}")
            # plt.savefig(self.output_dir / utils.unique_filename("png", "latency_test"), bbox_inches="tight")
            plt.savefig(self.output_dir / f"predict_{latency_predictor_type}_testsplit.png", bbox_inches="tight")

            l_test, l_pred, l_losses = latency_predictor.test(train_dataset, gpu_id=self.gpu_id)
            df = pd.DataFrame({"timeloop": l_test.flatten(), "model": l_pred.flatten()})
            # df.to_csv(self.output_dir / "latency_train.csv")
            # df = df[df['timeloop'] <= 1e9]
            plt.figure()
            plt.scatter(df["timeloop"], df["model"], s=0.5)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(10, 10e10)
            plt.ylim(10, 10e10)
            r2 = sklearn.metrics.r2_score(df["timeloop"], df["model"])
            sc = scipy.stats.spearmanr(df["timeloop"], df["model"]).correlation
            plt.xlabel("real (cycles)")
            plt.ylabel("pred (cycles)")
            plt.title(f"Latency Predictions (Test Set), corr.={round(sc,2)}")
            # plt.savefig(self.output_dir / utils.unique_filename("png", "latency_test"), bbox_inches="tight")
            plt.savefig(self.output_dir / f"predict_{latency_predictor_type}_trainsplit.png", bbox_inches="tight")

            # REBUTTAL
            l_test, l_pred, l_losses = latency_predictor.test(extra_test_data, gpu_id=self.gpu_id, num_unplot_points=len(test_dataset)//2*0)
            df = pd.DataFrame({"timeloop": l_test.flatten(), "model": l_pred.flatten()})
            df.to_csv(self.output_dir / "latency_gemmini_test.csv")
            # df = df[df['timeloop'] <= 1e9]
            plt.figure()
            plt.scatter(df["timeloop"], df["model"], s=0.5)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("real (cycles)")
            plt.ylabel("pred (cycles)")
            # plt.xlim(10, 10e10)
            # plt.ylim(10, 10e10)
            r2 = sklearn.metrics.r2_score(df["timeloop"], df["model"])
            sc = scipy.stats.spearmanr(df["timeloop"], df["model"]).correlation
            plt.title(f"Latency Predictions (DOSA-Searched Points), corr.={round(sc,2)}")
            # plt.savefig(self.output_dir / utils.unique_filename("png", "latency_test"), bbox_inches="tight")
            plt.savefig(self.output_dir / f"predict_{latency_predictor_type}_dosa.png", bbox_inches="tight")
            exit(0)

        # area_model = predictors.train_mlp(train_dataset, ["arch"], ["target.area"], hidden_layer_sizes = (81,27,9),
        #                                   output_dir=self.output_dir, save_str="conv_arch", num_iters=10000, gpu_id=self.gpu_id,
        #                                   continue_training=True)
        # arch_keys = utils.keys_by_type(train_dataset.df, "arch")
        # area_test_x = pytorch_util.from_numpy(test_dataset.df[arch_keys].to_numpy())
        # area_pred = area_model(area_test_x)
        # import pdb
        # pdb.set_trace()
        # a_test = train_dataset.denorm("target.area", test_dataset.df["target.area"].to_numpy().flatten())
        # a_pred = train_dataset.denorm("target.area", pytorch_util.to_numpy(area_pred.detach()).flatten())
        # df = pd.DataFrame({"timeloop": a_test.flatten(), "model": a_pred.flatten()})
        # df.to_csv(self.output_dir / "area_test.csv")
        # plt.scatter(a_test, a_pred, s=0.3)
        # plt.xlabel("real (mm^2)")
        # plt.ylabel("pred (mm^2)")
        # plt.title("Area Predictions (Test Set)")
        # plt.savefig(self.output_dir / utils.unique_filename("png", "area_test"), bbox_inches="tight")
        # exit(0)
        
        mapping_min_hw, _, _ = min_hw(self.arch_name, self.output_dir, [(pytorch_util.to_numpy(flat_mapping), prob,)])
        mapping_min_hw = train_dataset.norm("arch", mapping_min_hw)
        logger.info("normed hw config: %s", mapping_min_hw)
        example_prob = []
        prob_keys = utils.keys_by_type(train_dataset.df, "prob")
        for key in prob_keys:
            dim = key[5:] # remove "prob."
            example_prob.append(prob.prob[dim])
        example_prob = pytorch_util.from_numpy(np.array(example_prob))
        example_prob = train_dataset.norm("prob", example_prob)

        layers_tensor = []
        prob_keys = utils.keys_by_type(train_dataset.df, "prob")
        for prob in self.layers:
            flat_layer = []
            for key in prob_keys:
                dim = key[5:] # remove "prob."
                flat_layer.append(prob.prob[dim])
            layers_tensor.append(flat_layer)
        layers_tensor = pytorch_util.from_numpy(np.array(layers_tensor))
        layers_tensor = train_dataset.norm("prob", layers_tensor)
        layers_tensor.requires_grad = False

        access_means = []
        for access_key in access_keys:
            access_mean = stats[access_key + "_max"]
            if access_mean == 0:
                access_means.append(1)
            else:
                access_means.append(access_mean)
        access_means = pytorch_util.from_numpy(np.array(access_means))

        # # MICRO REBUTTAL: Interpolation study
        # arch_name = "gemmini"
        # hw_config = [16, 128, 32]
        # prob = Prob(pathlib.Path("/scratch/charleshong/dla-dataset/dataset/workloads/resnet50/_outputs_input.70.yaml"))
        # output_dir = pathlib.Path("/scratch/charleshong/dla-dataset/output_dir_interpolation")
        # arch_config = init_hw_config(arch_name, hw_config, output_dir)
        # row = arch_config.run_cosa(prob, run_mapping=True)[0]
        # energy = row["target.energy"]
        # cycle = row["target.cycle"]
        # logger.debug("Energy: %s, Cycle: %s", energy, cycle)
        # flat_mapping = list(mapping_utils.process_mapping(row["mapping.mapping"], prob.shape))
        # interp_layers_tensor = []
        # prob_keys = utils.keys_by_type(train_dataset.df, "prob")
        # flat_layer = []
        # for key in prob_keys:
        #     dim = key[5:] # remove "prob."
        #     flat_layer.append(prob.prob[dim])
        # interp_layers_tensor = pytorch_util.from_numpy(np.array([flat_layer]))
        # interp_layers_tensor = train_dataset.norm("prob", interp_layers_tensor)

        # # Interpolate
        # dim = "C"
        # step = 1
        # P_vals = []
        # interp_edps = []
        # for P_val in np.arange(1, prob.prob[dim] + step, step):
        #     flat_mapping_interp = flat_mapping.copy()
        #     flat_mapping_interp = pytorch_util.from_numpy(np.array(flat_mapping_interp))
        #     P_0_idx = mapping_utils.mapping_index(4, 7, 0, "temporal", prob.prob_name_idx_dict[dim])
        #     P_1_idx = mapping_utils.mapping_index(4, 7, 1, "temporal", prob.prob_name_idx_dict[dim])
        #     P_3_idx = mapping_utils.mapping_index(4, 7, 3, "temporal", prob.prob_name_idx_dict[dim])
        #     spatial_factor = 1
        #     if dim == "C":
        #         spatial_idx = mapping_utils.mapping_index(4, 7, 1, "spatial", prob.prob_name_idx_dict[dim])
        #     #     spatial_factor = float(flat_mapping_interp[spatial_idx])
        #     flat_mapping_interp[P_0_idx] = 1
        #     # flat_mapping_interp[P_1_idx] = P_val / spatial_factor
        #     flat_mapping_interp[P_1_idx] = prob.prob[dim] / (spatial_factor * P_val)
        #     flat_mapping_interp[spatial_idx] = P_val
        #     flat_mapping_interp[P_3_idx] = 1
        #     min_hw_config, cap_per_layer, max_cap_idxs = min_hw(arch_name, output_dir, [(flat_mapping_interp, prob)], grad=True)
        #     flat_mapping_interp = flat_mapping_interp.unsqueeze(0)
        #     min_hw_config_normalized = train_dataset.norm("arch", min_hw_config)
        #     relevant_accesses = self.get_accesses(flat_mapping_interp, [prob])
        #     normed_relevant_accesses = relevant_accesses / access_means
        #     interp_latency_pred = latency_predictor.predict(min_hw_config_normalized, train_dataset.norm("mapping", flat_mapping_interp)[:, relevant_idxs], normed_relevant_accesses, interp_layers_tensor)
        #     interp_latency_pred_denormed = train_dataset.denorm("target.cycle", interp_latency_pred).sum().item()
        #     interp_energy_pred = energy_predictor.predict(min_hw_config, normed_relevant_accesses)
        #     interp_energy_pred_denormed = energy_predictor.denorm_energy(interp_energy_pred).sum().item()
        #     interp_edp = interp_latency_pred_denormed * interp_energy_pred_denormed
        #     logger.error("P_val: %s, interp_edp: %s", P_val / spatial_factor, interp_edp)
        #     if P_val >= 512:
        #         import pdb
        #         pdb.set_trace()
        #     if P_val / spatial_factor < 1:
        #         continue
        #     P_vals.append(P_val / spatial_factor)
        #     interp_edps.append(interp_edp)
        # df = pd.DataFrame.from_dict({"dim": P_vals, "EDP": interp_edps})
        # df.to_csv("/scratch/charleshong/dla-dataset/data/interpolation.csv")
        # exit(0)

        results = list()
        hw_config_lst = list()
        mappings_lst = list()
        losses_lst = list()
        steps_lst = list()
        start_arch_configs_used = list()
        dfs = list()
        start_perf = float("inf")
        opt_path = self.output_dir / utils.unique_filename("pt", "gd_opt")
        layer_count_gpu = pytorch_util.from_numpy(np.array(self.layer_count)).unsqueeze(-1)
        lowest_first_iter_perf_loss = float('inf')
        num_tries_start = 0
        while len(results) < num_starts:
            num_tries_start += 1
            num_iters = base_gd_iters + per_start_gd_step * len(results)
            steps_lst.append(num_iters)

            logger.info("Searching start %s of %s", len(results)+1, num_starts)
            logger.info("Initializing CoSA mappings for %s layers", len(self.layers))
            if num_tries_start - 1 < len(random_arch_configs):
                arch_config = random_arch_configs[num_tries_start - 1]
            else:
                arch_config = init_hw_config(self.arch_name, "random", self.output_dir)
                random_arch_configs.append(arch_config)
            procs = []
            for prob in self.layers:
                proc = arch_config.run_cosa(prob, run_mapping=True, run_async=True)
                procs.append(proc)
            running_procs = [True for _ in procs]

            mappings = [None] * len(self.layers)
            energy_total = 0
            cycle_total = 0
            while any(running_procs):
                for i in range(len(procs)):
                    if not running_procs[i]:
                        continue
                    proc = procs[i]
                    retcode = proc.poll()
                    if retcode is not None:
                        rows = arch_config.collect_cosa_result(self.layers[i], run_mapping=True)
                        row = rows[0]
                        flat_mapping = list(mapping_utils.process_mapping(row["mapping.mapping"], self.layers[i].shape))

                        # # real gemmini mapping: fix loop ordering
                        # reg_perm_idx = mapping_utils.mapping_index(4, 7, 0, "perm", 0)
                        # acc_perm_idx = mapping_utils.mapping_index(4, 7, 1, "perm", 0)
                        # flat_mapping[reg_perm_idx:reg_perm_idx+7] = [3, 4, 1, 2, 5, 6, 7] # PQRSCKN
                        # flat_mapping[acc_perm_idx:acc_perm_idx+7] = [5, 6, 1, 2, 4, 7, 3] # PQNCRSK

                        energy_total += row["target.energy"] * self.layer_count[i]
                        cycle_total += row["target.cycle"] * self.layer_count[i]
                        mappings[i] = flat_mapping
                        running_procs[i] = False
                time.sleep(0.5)
            logger.debug("Total cycle %s, total energy %s, EDP %s", cycle_total, energy_total, cycle_total*energy_total)
            if self.metric == "edp":
                cosa_result = cycle_total*energy_total
            elif self.metric == "energy":
                cosa_result = energy_total
            # logger.debug("Orig CoSA mappings: %s", mappings)
            if len(results) == 0:
                start_perf = cosa_result

            logger.info("Setting tensor requires_grad")
            normed_mappings = train_dataset.norm("mapping", mappings)#[:,relevant_idxs]
            # normed_dram_part = pytorch_util.from_numpy(np.array(normed_mappings)[:,:14])
            if ordering_search_method == "gumbel" or ordering_search_method == "softmax":
                input = pytorch_util.from_numpy(np.array(normed_mappings)[:,21:])
            else:
                input = pytorch_util.from_numpy(np.array(normed_mappings)[:,14:])
            input.requires_grad = True
            # logger.debug("Orig CoSA mapping (normed): %s", input)

            logger.debug("Number of GD steps before rounding: %s", round_steps)
            logger.info("Running optimization for %s iters", num_iters)
            losses = []
            # optimizer = torch.optim.SGD([input], lr=sgd_lr, momentum=0.9)
            optimizer = torch.optim.Adam([input], lr=sgd_lr)

            dram_filler = pytorch_util.from_numpy(np.zeros((len(self.layers), 14)))
            if ordering_search_method == "gumbel" or ordering_search_method == "softmax":
                # dram_filler = torch.cat((dram_filler, pytorch_util.from_numpy(np.array(normed_mappings)[:,14:21])), dim=1)
                dram_filler = torch.cat((dram_filler, pytorch_util.from_numpy(np.ones((len(self.layers), 7))*7)), dim=1)

            retry_start = False
            min_rounded_pred = float('inf')
            # logger.setLevel(logging.ERROR)
            for iter in range(num_iters):
                if self.log_times:
                    start_time = time.time()
                full_mappings = torch.cat((dram_filler, input), dim=1)
                denormed_full_mapping = train_dataset.denorm("mapping", full_mappings)
                # logger.info("dram perm: %s", denormed_full_mapping[:,14:21])
                dram_part = self.get_dram_factors(denormed_full_mapping, relevant_keys)
                denormed_full_mapping = torch.cat((dram_part, denormed_full_mapping[:,14:]), dim=1)

                # logger.debug("denormed input pre-clamp: %s", denormed_full_mapping)
                invalid_mapping_loss = torch.square(torch.clamp(1 - denormed_full_mapping, min=0)).sum()
                # invalid_mapping_loss = invalid_mapping_loss + torch.clamp(1 - denormed_full_mapping, min=-0.1, max=0).sum() * 0.01
                # logger.debug("denormed input post-clamp: %s", denormed_full_mapping)
                mapping_prob_pairs = [(denormed_full_mapping[i], self.layers[i]) for i in range(len(self.layers))]
                hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs, grad=True)
                hw_config_normalized = train_dataset.norm("arch", hw_config)

                # use for firesim experiments
                if constant_pe:
                    hw_config_loss = torch.square(torch.clamp(hw_config[0] - 16, min=0)) * 10
                else:
                    hw_config_loss = 0

                full_mappings = train_dataset.norm("mapping", denormed_full_mapping)
                # normed_dram_part = full_mappings[:,:14]

                if self.log_times:
                    logger.info("dram factor time %s", time.time() - start_time)
                
                if self.log_times:
                    pred_start_time = time.time()

                ##### Area and/or energy prediction
                # area_pred = area_model(hw_config_normalized)
                relevant_accesses = self.get_accesses(denormed_full_mapping, self.layers)
                normed_relevant_accesses = relevant_accesses / access_means

                if iter == 0:
                    first_iter_energy_pred = energy_predictor.predict(hw_config, normed_relevant_accesses) * layer_count_gpu
                    first_iter_latency_pred = latency_predictor.predict(hw_config_normalized, full_mappings[:,relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                    first_iter_perf_loss = self.get_perf_loss(first_iter_latency_pred, first_iter_energy_pred).detach()
                    if first_iter_perf_loss > lowest_first_iter_perf_loss * 10:
                        logger.debug("Lowest first_iter_perf_loss %s, predicted first_iter_perf_loss %s, skipping this start point",
                                     lowest_first_iter_perf_loss, first_iter_perf_loss)
                        retry_start = True
                        break
                    elif first_iter_perf_loss < lowest_first_iter_perf_loss:
                        logger.debug("New lowest starting first_iter_perf_loss %s", first_iter_perf_loss)
                        lowest_first_iter_perf_loss = first_iter_perf_loss

                if ordering_search_method == "gumbel" or ordering_search_method == "softmax":
                    layer_latency_preds = []
                    layer_energy_preds = []
                    for l in range(len(self.layers)):
                        latency_preds = []
                        energy_preds = []
                        perf_losses = []
                        for perm in self.gen_all_perms():
                            denormed_full_mapping_l = torch.cat((denormed_full_mapping[l][:14], perm, denormed_full_mapping[l][21:]))
                            layer_relevant_accesses = self.get_accesses([denormed_full_mapping_l], [self.layers[l]])
                            normed_layer_relevant_accesses = layer_relevant_accesses / access_means
                            perm_latency_pred = latency_predictor.predict(hw_config_normalized,
                                                                     train_dataset.norm("mapping", denormed_full_mapping_l)[relevant_idxs].unsqueeze(0), 
                                                                     normed_layer_relevant_accesses,
                                                                     layers_tensor[l].unsqueeze(0)
                                                                    ) * layer_count_gpu[l]
                            perm_energy_pred = energy_predictor.predict(hw_config, normed_layer_relevant_accesses) * layer_count_gpu[l]
                            perm_perf_loss = self.get_perf_loss(perm_latency_pred, perm_energy_pred)
                            latency_preds.append(perm_latency_pred)
                            energy_preds.append(perm_energy_pred)
                            perf_losses.append(perm_perf_loss.unsqueeze(0))

                        latency_preds = torch.cat(latency_preds)
                        energy_preds = torch.cat(energy_preds)
                        perf_losses = torch.cat(perf_losses)
                        perf_losses_inverted = 1 / perf_losses
                        perm_logits = perf_losses_inverted / perf_losses_inverted.max()

                        if ordering_search_method == "gumbel":
                            gumbel_output = torch.nn.functional.gumbel_softmax(torch.log(perm_logits), tau=gumbel_temp, hard=False, dim=-1)
                        elif ordering_search_method == "softmax":
                            gumbel_output = torch.nn.functional.softmax(perf_losses_inverted, dim=-1)
                        else:
                            logger.error("Invalid ordering search method %s", ordering_search_method)
                        # gumbel_output = perf_losses_inverted / perf_losses_inverted.sum()

                        # logger.debug("gumbel_output: %s", gumbel_output)

                        # flow_map = {
                        #     0: "WS",
                        #     1: "IS",
                        #     2: "OS",
                        # }
                    
                        # with open(self.output_dir / "gumbel_output.txt", "a") as f:
                        #     perm = denormed_full_mapping[l,14:21]
                        #     flow = flow_map[gumbel_output.argmax().item()]
                        #     write_str = flow
                        #     if l != len(self.layers) - 1:
                        #         write_str += ","
                        #     f.write(write_str)

                        layer_latency_pred = (gumbel_output * latency_preds.squeeze()).sum().unsqueeze(0)
                        layer_energy_pred = (gumbel_output * energy_preds.squeeze()).sum().unsqueeze(0)
                        layer_latency_preds.append(layer_latency_pred)
                        layer_energy_preds.append(layer_energy_pred)

                    # with open(self.output_dir / "gumbel_output.txt", "a") as f:
                    #     f.write("\n")

                    layer_latency_preds = torch.cat(layer_latency_preds).unsqueeze(-1)
                    layer_energy_preds = torch.cat(layer_energy_preds).unsqueeze(-1)
                    latency_pred = layer_latency_preds * layer_count_gpu
                    energy_pred = layer_energy_preds * layer_count_gpu
                else:
                    # import pdb
                    # pdb.set_trace()
                    # [4, 5, 1, 2, 6, 7, 3], # PQNRSCK, weight stationary
                    # [2, 3, 4, 5, 6, 1, 7], # KRSPQCN, input stationary
                    # [1, 2, 5, 6, 3, 4, 7], # RSCKPQN, output stationary

                    # stationarity_map = {
                    #     0: "WS",
                    #     1: "IS",
                    #     2: "OS",
                    # }
                    # with open(self.output_dir / "gumbel_no.txt", "a") as f:
                    #     for l in range(len(self.layers)):
                    #         perm = denormed_full_mapping[l,14:21]
                    #         flow = ""
                    #         if perm[2] < perm[4]: # P inner to C
                    #             if perm[2] < perm[5]: # P inner to K and C
                    #                 flow = "WS"
                    #             else: # K inner to P and C
                    #                 flow = "IS"
                    #         else:
                    #             if perm[5] < perm[4]: # K inner to C and P
                    #                 flow = "IS"
                    #             else: # C inner to K and P
                    #                 flow = "OS"
                    #         write_str = flow
                    #         if l != len(self.layers) - 1:
                    #             write_str += ","
                    #         f.write(write_str)
                    #     f.write("\n")

                    # normed_relevant_accesses = train_dataset.norm("dse.access", relevant_accesses)
                    energy_pred = energy_predictor.predict(hw_config, normed_relevant_accesses) * layer_count_gpu

                    # latency_pred = latency_model(torch.cat((full_mappings[:,relevant_idxs], layers_tensor), dim=1))
                    latency_pred = latency_predictor.predict(hw_config_normalized, full_mappings[:,relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                    # logger.debug("relevant_accesses: %s", relevant_accesses)
                    # logger.debug("normed_relevant_accesses: %s", normed_relevant_accesses)
                    # logger.debug("normed_latency_pred: %s", latency_pred)
                    # latency_pred = latency_predictor.predict(full_mappings[:,relevant_idxs], layers_tensor)

                denormed_energy_pred = energy_predictor.denorm_energy(energy_pred)
                if denormed_energy_pred.sum() < 0:
                    logger.debug("relevant_accesses: %s", relevant_accesses)
                    logger.debug("normed_relevant_accesses: %s", normed_relevant_accesses)
                    logger.debug("access count coefficients: %s", energy_predictor.predict_coeff(hw_config))

                if self.log_times:
                    logger.info("perf pred time %s", time.time() - pred_start_time)
                    bprop_start_time = time.time()

                # # MICRO REBUTTAL: get performance of rounded mappings
                # rounded_denormed_full_mapping = [mapping_utils.round_mapping(pytorch_util.to_numpy(denormed_full_mapping[i]), self.layers[i], round_perms=False) for i in range(len(self.layers))]
                # rounded_denormed_full_mapping = pytorch_util.from_numpy(np.array(rounded_denormed_full_mapping)).detach()
                # mapping_prob_pairs = [(rounded_denormed_full_mapping[i], self.layers[i]) for i in range(len(self.layers))]
                # rounded_hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs, grad=True)
                # rounded_hw_config_normalized = train_dataset.norm("arch", rounded_hw_config)
                # relevant_accesses = self.get_accesses(rounded_denormed_full_mapping, self.layers)
                # normed_relevant_accesses = relevant_accesses / access_means
                # rounded_latency_pred = latency_golden.predict(rounded_hw_config_normalized, train_dataset.norm("mapping", rounded_denormed_full_mapping)[:, relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                # rounded_latency_pred_denormed = train_dataset.denorm("target.cycle", rounded_latency_pred).sum().item()
                # rounded_energy_pred = energy_golden.predict(rounded_hw_config, normed_relevant_accesses) * layer_count_gpu
                # rounded_energy_pred_denormed = energy_golden.denorm_energy(rounded_energy_pred).sum().item()
                # rounded_edp = rounded_latency_pred_denormed * rounded_energy_pred_denormed
                # unrounded_edp = train_dataset.denorm("target.cycle", latency_pred).sum().item() * denormed_energy_pred.sum().item()
                # logger.error("step: %s, rounded latency: %s, rounded energy: %s, rounded EDP: %s, unrounded EDP: %s", iter, rounded_latency_pred_denormed, rounded_energy_pred_denormed, rounded_edp, unrounded_edp)

                # fake_targets = [latency_pred, 0, 0, area_pred] # cycle, energy, edp, area
                # fake_targets = train_dataset.denorm("target", fake_targets)
                # loss = torch.exp2(latency_pred*latency_std+latency_mean).sum() * torch.exp2(area_pred*area_std+area_mean)
                # loss = (torch.exp2(latency_pred*latency_std+latency_mean) / (2**latency_mean)).sum() * denormed_energy_pred / ((2**energy_mean)*len(self.layers))
                # loss = (torch.log2((torch.exp2(latency_pred-100)).sum()) + 100) * latency_std + torch.log2(denormed_energy_pred) # / ((2**energy_mean)*len(self.layers)))
                # loss = denormed_energy_pred

                perf_loss = self.get_perf_loss(latency_pred, energy_pred)

                # logger.debug("latency part %s", torch.log2((torch.exp2(latency_pred)).sum()) * latency_std)
                # loss = (latency_pred*latency_std+latency_mean).sum() * torch.log2(denormed_energy_pred)
                # arch_loss = torch.clamp(hw_config_normalized, min=0).sum()
                arch_loss = 0
                loss = perf_loss + 0.2*abs(perf_loss.item())*arch_loss
                loss = loss + abs(perf_loss.item())*invalid_mapping_loss
                loss = loss + abs(perf_loss.item())*hw_config_loss
                if torch.isinf(loss) or torch.isnan(loss):
                    import pdb
                    pdb.set_trace()
                losses.append(loss.item())
                # logger.debug("Predicted min hw (denormed): %s, min hw (total denormed): %s, predicted area (denormed) %s, predicted latency (denormed): %s, predicted ADP %s", 
                #              pytorch_util.to_numpy(train_dataset.denorm("arch", min_hw_preds)).flatten(), pytorch_util.to_numpy(train_dataset.denorm("arch", hw_config_normalized)), 
                #              torch.exp2(area_pred*area_std+area_mean).item(), torch.exp2(latency_pred*latency_std+latency_mean).sum().item(), loss.item())
                logger.debug("%s Min hw (total denormed): %s, predicted energy (denormed) %s, predicted latency (denormed): %s, predicted EDP %s, arch regularization loss %s, invalid mapping loss %s, total loss %s",
                             iter, hw_config, denormed_energy_pred.sum(), train_dataset.denorm("target.cycle", latency_pred).sum().item(), train_dataset.denorm("target.cycle", latency_pred).sum() * denormed_energy_pred.sum(),
                             0.2*abs(perf_loss.item())*arch_loss, abs(perf_loss.item())*invalid_mapping_loss, loss)
                
                # wandb.log({f"start-{len(results)}": {
                #     "latency-pred": train_dataset.denorm("target.cycle", latency_pred).sum().item(),
                #     "energy-pred": denormed_energy_pred.sum().item(),
                #     "edp-pred": train_dataset.denorm("target.cycle", latency_pred).sum().item() * denormed_energy_pred.sum().item(),
                #     "perf-loss": perf_loss,
                #     "arch-loss": arch_loss,
                #     "invalid-mapping-loss": invalid_mapping_loss,
                #     "loss": loss,
                # }})

                # after rounding, save mapping if it has best predicted perf so far
                if ((iter+1) % round_steps) == 1:
                    denormed_latency_pred = train_dataset.denorm("target.cycle", latency_pred)
                    logger.debug("denormed_latency_pred %s", denormed_latency_pred)
                    if perf_loss < min_rounded_pred:
                        min_rounded_pred = perf_loss
                        min_rounded_mapping = denormed_full_mapping.clone().detach()
                        best_pred_iter = iter

                # logger.debug("Normed mapping: %s", input)
                # logger.debug("Denormed mapping: %s", train_dataset.denorm("mapping", input))
                # example_mappings[:,relevant_idxs] = pytorch_util.to_numpy(input)
                # this_iter_result = self.eval_mappings(np.array(train_dataset.denorm("mapping", example_mappings)))
                optimizer.zero_grad()
                try:
                    loss.backward(inputs=(input,))
                except:
                    logger.debug(traceback.format_exc())
                    continue
                # torch.nn.utils.clip_grad_norm_(input, 100)
                optimizer.step()

                if ((iter+1) % round_steps) == 0 or iter == num_iters - 1:
                    logger.debug("Predicted latency per layer (before rounding): %s", train_dataset.denorm("target.cycle", latency_pred))
                    logger.debug("Predicted energy per layer (before rounding): %s", denormed_energy_pred)
                    logger.debug("Predicted total latency (before rounding): %s", (train_dataset.denorm("target.cycle", latency_pred) * layer_count_gpu).sum())
                    logger.debug("Predicted total energy (before rounding): %s", (denormed_energy_pred * layer_count_gpu).sum())

                    # every N iters, round to the nearest valid mapping
                    full_mappings = torch.cat((dram_filler, input), dim=1)
                    denormed_full_mapping = train_dataset.denorm("mapping", full_mappings)

                    dram_factors = self.get_dram_factors(denormed_full_mapping, relevant_keys)
                    denormed_full_mapping = torch.cat((dram_factors, denormed_full_mapping[:,14:]), dim=1)

                    denormed_full_mapping = pytorch_util.to_numpy(denormed_full_mapping)
                    # logger.debug("denormed_full_mapping: %s", torch.tensor(denormed_full_mapping))
                    rounded_denormed_full_mapping = [mapping_utils.round_mapping(denormed_full_mapping[i], self.layers[i], round_perms=False) for i in range(len(self.layers))]
                    # logger.debug("rounded_denormed_full_mapping: %s", torch.tensor(np.array(rounded_denormed_full_mapping)))

                    rounded_denormed_full_mapping = pytorch_util.from_numpy(np.array(rounded_denormed_full_mapping)).detach()
                    mapping_prob_pairs = [(rounded_denormed_full_mapping[i], self.layers[i]) for i in range(len(self.layers))]
                    hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs, grad=True)
                    hw_config_normalized = train_dataset.norm("arch", hw_config)
                    relevant_accesses = self.get_accesses(rounded_denormed_full_mapping, self.layers)
                    normed_relevant_accesses = relevant_accesses / access_means
                    latency_pred = latency_predictor.predict(hw_config_normalized, train_dataset.norm("mapping", rounded_denormed_full_mapping)[:, relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                    energy_pred = energy_predictor.predict(hw_config, normed_relevant_accesses) * layer_count_gpu
                    denormed_energy_pred = energy_predictor.denorm_energy(energy_pred)

                    logger.debug("Predicted latency per layer (after rounding): %s", train_dataset.denorm("target.cycle", latency_pred))
                    logger.debug("Predicted energy per layer (after rounding): %s", denormed_energy_pred)
                    logger.debug("Predicted total latency (after rounding): %s", (train_dataset.denorm("target.cycle", latency_pred) * layer_count_gpu).sum())
                    logger.debug("Predicted total energy (after rounding): %s", (denormed_energy_pred * layer_count_gpu).sum())

                    if ordering_search_method == "gumbel" or ordering_search_method == "softmax":
                        best_perf = float("inf")
                    else:
                        best_perf = self.get_perf_loss(latency_pred, energy_pred)

                    if ordering_search_method != "none":
                        logger.debug("Perf before trying perms: %s %s %s", latency_pred, energy_pred, best_perf)
                        # try all perms for dram level
                        for l in range(len(self.layers)):
                            relevant_accesses_clone = relevant_accesses.clone()
                            best_perm = rounded_denormed_full_mapping[l][14:21].clone()
                            best_accesses = relevant_accesses[l]
                            for perm in self.gen_all_perms():
                                # random_perm = self.gen_random_perm(self.layers[l])
                                rounded_denormed_full_mapping[l][14:21] = perm
                                mapping_prob_pairs = [(rounded_denormed_full_mapping[i], self.layers[i]) for i in range(len(self.layers))]
                                hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs, grad=True)
                                hw_config_normalized = train_dataset.norm("arch", hw_config)
                                layer_relevant_accesses = self.get_accesses([rounded_denormed_full_mapping[l]], [self.layers[l]])
                                relevant_accesses_clone[l] = layer_relevant_accesses
                                normed_relevant_accesses = relevant_accesses_clone / access_means
                                latency_pred = latency_predictor.predict(hw_config_normalized, train_dataset.norm("mapping", rounded_denormed_full_mapping)[:, relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                                energy_pred = energy_predictor.predict(hw_config, normed_relevant_accesses) * layer_count_gpu
                                perf_loss = self.get_perf_loss(latency_pred, energy_pred)
                                if perf_loss < best_perf:
                                    best_perf = perf_loss
                                    best_perm = perm
                                    best_accesses = layer_relevant_accesses
                                    logger.debug("Layer %s new best perm perf %s", l, best_perf)
                            rounded_denormed_full_mapping[l][14:21] = best_perm
                            relevant_accesses[l] = best_accesses
                        
                        mapping_prob_pairs = [(rounded_denormed_full_mapping[i], self.layers[i]) for i in range(len(self.layers))]
                        hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs, grad=True)
                        hw_config_normalized = train_dataset.norm("arch", hw_config)
                        relevant_accesses = self.get_accesses(rounded_denormed_full_mapping, self.layers)
                        normed_relevant_accesses = relevant_accesses / access_means
                        latency_pred = latency_predictor.predict(hw_config_normalized, train_dataset.norm("mapping", rounded_denormed_full_mapping)[:, relevant_idxs], normed_relevant_accesses, layers_tensor) * layer_count_gpu
                        energy_pred = energy_predictor.predict(hw_config, normed_relevant_accesses) * layer_count_gpu
                        logger.debug("Perf after trying perms: %s %s %s", latency_pred, energy_pred, self.get_perf_loss(latency_pred, energy_pred))

                    if iter == num_iters - 1:
                        perf_loss = best_perf
                        # wandb.log({f"start-{len(results)}": {
                        #     "latency-pred": train_dataset.denorm("target.cycle", latency_pred).sum().item(),
                        #     "energy-pred": denormed_energy_pred.sum().item(),
                        #     "edp-pred": train_dataset.denorm("target.cycle", latency_pred).sum().item() * denormed_energy_pred.sum().item(),
                        #     "perf-loss": perf_loss,
                        # }})
                        if perf_loss < min_rounded_pred:
                            min_rounded_pred = perf_loss
                            min_rounded_mapping = rounded_denormed_full_mapping.clone().detach()
                            best_pred_iter = iter
                        break

                    if ordering_search_method == "gumbel" or ordering_search_method == "softmax":
                        input = train_dataset.norm("mapping", rounded_denormed_full_mapping)[:,21:]
                    else:
                        input = train_dataset.norm("mapping", rounded_denormed_full_mapping)[:,14:]

                    input.requires_grad=True
                    # optimizer = torch.optim.SGD([input], lr=sgd_lr, momentum=0.9)
                    # sdg_lr = sgd_lr / 2
                    torch.save(optimizer.state_dict(), opt_path)
                    optimizer = torch.optim.Adam([input], lr=sgd_lr)
                    optimizer.load_state_dict(torch.load(opt_path))

                # logger.info("denormed mappings: %s", denormed_input)
                # input = pytorch_util.from_numpy(np.array(train_dataset.norm("mapping", denormed_input))[:,relevant_idxs])
                # logger.info("normed mappings: %s", input)
                # input.requires_grad = True

                if self.log_times:
                    logger.info("bprop time %s", time.time() - bprop_start_time)
                    logger.info("total iter time %s", time.time() - start_time)

            if retry_start:
                continue

            logger.debug("Using mappings from iter %s", best_pred_iter)
            mappings = pytorch_util.to_numpy(min_rounded_mapping)
            mapping_prob_pairs = [(mappings[i], self.layers[i],) for i in range(len(self.layers))]
            hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
            mappings_lst.append(mappings)
            hw_config_lst.append(hw_config)
            start_arch_configs_used.append(arch_config)
            logger.info("Found mappings %s, hw config %s", mappings.tolist(), hw_config)

            if self.metric == "edp":
                denormed_perf_loss = latency_predictor.denorm_cycle(energy_predictor.denorm_energy(min_rounded_pred))
            if self.metric == "energy":
                denormed_perf_loss = energy_predictor.denorm_energy(min_rounded_pred)
            losses_lst.append(denormed_perf_loss)
            logger.debug("predicted loss: %s", denormed_perf_loss)
            try:
                rows = self.eval_mappings(mappings, return_rows=True)
                df = pd.DataFrame(rows)
                dfs.append(df)
                df_path = self.output_dir / utils.unique_filename("csv", f"gd_results_{self.workload_name}_{len(results)}")
                logger.debug("Saving rows to %s", df_path)
                df.to_csv(df_path, index=False)
                this_start_result = self.aggregate_rows(rows)
                results.append(this_start_result)
                # wandb.log({"results": this_start_result,
                #            "cosa-results": cosa_result})
            except:
                logger.warning(traceback.format_exc())
                logger.warning("Could not evaluate mappings for start %s", len(results))
                results.append(results[-1])

        # for p in latency_model.parameters():
        #     p.requires_grad = True
        # for p in area_model.parameters():
        #     p.requires_grad = True
        # for min_hw_model in min_hw_models:
        #     for p in min_hw_model.parameters():
        #         p.requires_grad = True
        idx = np.nanargmin(results)
        logger.debug("Searched for steps: %s", steps_lst)
        logger.info("Finished searching %s starts, results %s", num_starts, results)
        logger.info("Best result at idx %s, target value %s, hw %s, mappings %s",
                    idx, results[idx], hw_config_lst[idx], mappings_lst[idx])
        return idx, hw_config_lst, mappings_lst, results, losses_lst, start_arch_configs_used, dfs, start_perf

    def get_accesses(self, denormed_full_mapping, layers, with_cache=False):
        if self.log_times:
            access_start_time = time.time()

        relevant = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
        if with_cache:
            relevant.append([1,1,1])
        relevant_accesses = None
        for i in range(len(layers)):
            if self.log_times:
                accesses_count_start_time = time.time()
            reads, updates, writes = mapping_utils.accesses_from_mapping(denormed_full_mapping[i], layers[i], with_cache=with_cache)
            if self.log_times:
                logger.info("accesses count counting time %s", time.time() - accesses_count_start_time)
                accesses_add_start_time = time.time()
            r = reads[0][0]
            if isinstance(r, int):
                r = torch.tensor(r, dtype=torch.float32).to(pytorch_util.device)
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
                            r = torch.tensor(r, dtype=torch.float32).to(pytorch_util.device)
                        if isinstance(u, int):
                            u = torch.tensor(u, dtype=torch.float32).to(pytorch_util.device)
                        if isinstance(w, int):
                            w = torch.tensor(w, dtype=torch.float32).to(pytorch_util.device)
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
                # if mem_lvl == 1 or mem_lvl == 2:
                #     layer_accesses = torch.cat((layer_accesses, this_lvl_reads, this_lvl_writes))
                # else:
                #     # layer_accesses = torch.cat((layer_accesses, this_lvl_accesses))
                #     layer_accesses = torch.cat((layer_accesses, this_lvl_reads+this_lvl_writes))
            layer_accesses = layer_accesses.unsqueeze(0) # [[x, y, z, ...]]
            if relevant_accesses is None:
                relevant_accesses = layer_accesses
            else:
                relevant_accesses = torch.cat((relevant_accesses, layer_accesses), dim=0)
            if self.log_times:
                logger.info("accesses count adding time %s", time.time() - accesses_add_start_time)
        relevant_accesses = torch.clamp(relevant_accesses, 1)

        if self.log_times:
            logger.info("access count time %s", time.time() - access_start_time)

        return relevant_accesses

    def get_perf_loss(self, latency_pred: torch.Tensor, energy_pred: torch.Tensor, area_pred: torch.Tensor=None) -> torch.Tensor:
        if self.metric == "edp":
            log_once("loss_fn: latency_pred.sum() * energy_pred.sum()")
            # loss = torch.log2((latency_pred + latency_mean / latency_std).sum()) + torch.log2(denormed_energy_pred)
            # loss = torch.log2(train_dataset.denorm("target.cycle", latency_pred).sum()) + torch.log2(denormed_energy_pred)
            # perf_loss = (train_dataset.denorm("target.cycle", latency_pred).sum() / (latency_max * len(self.layers))) * (denormed_energy_pred / (energy_max * len(self.layers)))
            perf_loss = latency_pred.sum() * energy_pred.sum()
        elif self.metric == "energy":
            log_once("loss_fn: energy_pred.sum()")
            perf_loss = energy_pred.sum()
        return perf_loss

    def gen_random_perm(self, prob: Prob):
        num_dims = len(prob.prob_idx_name_dict)
        random_order = np.random.permutation(num_dims) + 1
        for idx, dim in prob.prob_idx_name_dict.items():
            if prob.prob[dim] == 1:
                random_order[idx] = num_dims+1
        return pytorch_util.from_numpy(random_order)

    def gen_all_perms(self): # TODO
        """RSPQCKN"""
        perms = [
            [4, 5, 1, 2, 6, 7, 3], # PQNRSCK, weight stationary
            [2, 3, 4, 5, 6, 1, 7], # KRSPQCN, input stationary
            [1, 2, 5, 6, 3, 4, 7], # RSCKPQN, output stationary
        ]
        return pytorch_util.from_numpy(np.array(perms))

    def get_dram_factors(self, denormed_mappings: torch.Tensor, relevant_keys: list):
        # set DRAM temporal tiling factors based on all rest of factors
        relevant_keys = set(relevant_keys)
        num_dims = 7
        num_mem_lvls = denormed_mappings.size(1) // num_dims // 3

        temporal_factors = []
        for dim_idx in range(num_dims):
            temporal_factors.append(self.prob_dims_recip[dim_idx].clone().unsqueeze(0))

        # TODO: construct prob dims tensor once and copy for each GD step

        for mem_lvl in range(0, num_mem_lvls-1):
            for dim_idx, dim in self.layers[0].prob_idx_name_dict.items():
                for fac_type in ("spatial", "temporal"):
                    key = f"mapping.{fac_type}_L{mem_lvl}_{dim}"
                    if key in relevant_keys:
                        mapping_idx = mapping_utils.mapping_index(num_mem_lvls, num_dims, mem_lvl, fac_type, dim_idx)
                        temporal_factors[dim_idx] = temporal_factors[dim_idx] * denormed_mappings[:,mapping_idx].unsqueeze(0)
        temporal_factors = torch.transpose(torch.cat(temporal_factors, dim=0), 0, 1)
        temporal_factors = 1 / (temporal_factors) # TODO: add epsilon to not create infs?

        spatial_factors = pytorch_util.from_numpy(np.array([[1.] * 7] * len(self.layers)))
        dram_factors = torch.cat((spatial_factors, temporal_factors), dim=1)
        return dram_factors

    def theoretical_min_cycles(self, pe_dim: int) -> int:
        mins = []
        for prob in self.layers:
            output_size = prob.prob["P"] * prob.prob["Q"]
            weight_size = prob.prob["R"] * prob.prob["S"]
            in_channel = prob.prob["C"]
            out_channel = prob.prob["K"]
            batch = prob.prob["N"]
            total_macs = output_size * weight_size * in_channel * out_channel * batch

            hw_macs = pe_dim ** 2 # TODO: update for new gemmini repr
            min_cycles = math.ceil(total_macs / hw_macs)
            mins.append(min_cycles)
        return mins

    def eval_mappings(self, mappings, return_rows=False):
        mapping_prob_pairs = [(mapping_utils.round_mapping(mappings[i], self.layers[i]), self.layers[i],) for i in range(len(self.layers))]
        hw_config, _, _ = min_hw(self.arch_name, self.output_dir, mapping_prob_pairs)
        arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
        rows = []
        for mapping, prob in mapping_prob_pairs:
            mapping_dict = arch_config.flat_mapping_to_dict(prob.shape, mapping)
            rows.append(arch_config.run_mapping_from_dict(prob, mapping_dict))
        latencies = [r["target.cycle"] for r in rows]
        energies = [r["target.energy"] for r in rows]
        logger.debug("Real latencies: %s", latencies)
        logger.debug("Real energies: %s", energies)
        if return_rows:
            return rows
        return self.aggregate_rows(rows)

    def select_prob_fn(self, mappings, hw_config, cap_per_layer, max_cap_idxs) -> int:
        # return max(max_cap_idxs, key=max_cap_idxs.count)
        
        max_per_idx = {}
        for idx in max_cap_idxs:
            max_per_idx[idx] = max_per_idx.get(idx, 0) + 1

        # First count how many levels (compute, then each mem lvl) each layer requires the most capacity for.
        # If a unique layer requires the most capacity at the most levels, choose that level.
        max_count = max(max_per_idx.values())
        max_count_idxs = [idx for idx in max_per_idx if max_per_idx[idx] == max_count]
        if len(max_count_idxs) == 1:
            return max_count_idxs[0]

        # If multiple layers are tied in how many levels are maximally utilized, 
        # choose the layer whose capacity at some level exceeds the mean by the greatest
        # number of standard deviations.
        cap_per_layer = np.array(cap_per_layer)
        max_diff = 0
        max_diff_idx = 0
        for lvl in range(cap_per_layer.shape[1]):
            mean = cap_per_layer[:, lvl].mean()
            std = cap_per_layer[:, lvl].std()
            diff = (cap_per_layer[max_cap_idxs[lvl], lvl] - mean) / std
            if diff > max_diff:
                max_diff = diff
                max_diff_idx = max_cap_idxs[lvl]
        return max_diff_idx

    def cosa_baseline(self, hw_config):
        arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
        procs = []
        for prob in self.layers:
            proc = arch_config.run_cosa(prob, run_mapping=True, run_async=True)
            procs.append(proc)
        running_procs = [True for _ in procs]

        rows = [None] * len(self.layers)
        while any(running_procs):
            for i in range(len(procs)):
                if not running_procs[i]:
                    continue
                proc = procs[i]
                retcode = proc.poll()
                if retcode is not None:
                    this_layer_rows = arch_config.collect_cosa_result(self.layers[i], run_mapping=True)
                    rows[i] = this_layer_rows[0]
                    running_procs[i] = False
            time.sleep(0.5)

        # arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
        # rows = []
        # for prob in self.layers:
        #     rows.extend(arch_config.run_cosa(prob))
        # logger.debug("CoSA baseline latencies: %s", [row["target.cycle"] for row in rows])
        # logger.debug("CoSA baseline energies: %s", [row["target.energy"] for row in rows])
        return self.aggregate_rows(rows)

    def random_baseline(self, hw_config, num_mappings):
        arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
        rows = []
        if self.metric == "edp":
            return_min_fn = lambda row: row["target.cycle"] * row["target.energy"]
        elif self.metric == "energy":
            return_min_fn = lambda row: row["target.energy"]
        for prob in self.layers:
            prob_rows = arch_config.run_random_mappings(prob, num_mappings, return_min_fn=return_min_fn)
            rows.extend(prob_rows)
        result = self.aggregate_rows(rows)
        return result

    def exhaustive_baseline(self, hw_config):
        arch_config = init_hw_config(self.arch_name, hw_config, self.output_dir)
        rows = []
        if self.metric == "edp":
            return_min_fn = lambda row: row["target.cycle"] * row["target.energy"]
        elif self.metric == "energy":
            return_min_fn = lambda row: row["target.energy"]
        for prob in self.layers:
            prob_rows = arch_config.run_exhaustive_mappings(prob, return_min_fn=return_min_fn)
            logger.info(prob_rows)
            rows.extend(prob_rows)
        result = self.aggregate_rows(rows)
        return result

    def aggregate_rows(self, rows) -> float:
        cycle = 0
        energy = 0
        area = 0
        for i in range(len(self.layers)):
            row = rows[i]
            layer_repeats = self.layer_count[i]
            cycle += row["target.cycle"] * layer_repeats
            energy += row["target.energy"] * layer_repeats
            area = row["target.area"]
        if self.metric == "edp":
            logger.debug("Real energy: %s, real latency: %s, real EDP: %s", energy, cycle, cycle*energy)
            return cycle * energy
        elif self.metric == "adp":
            logger.debug("Real area: %s, real latency: %s, real ADP: %s", area, cycle, cycle*area)
            return cycle * area
        elif self.metric == "energy":
            return energy

def add_col(s: pd.Series, arch_name, output_dir, prob_keys, dataset):
    mapping = s["mapping.flat_mapping"]
    prob_feats = list(s[prob_keys].values)
    prob_feats = dataset.denorm("prob", prob_feats)
    prob = eval.parse_prob(output_dir, prob_keys, prob_feats)
    mapping_prob_pairs = [(mapping, prob,)]
    hw_config_denormed, _, _ = min_hw(arch_name, output_dir, mapping_prob_pairs)
    hw_config = dataset.norm("arch", hw_config_denormed)
    # if list(mapping) == list([  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,  14., 256.,
    #      32.,   1.,   7.,   7.,   7.,   3.,   1.,   2.,   7.,   1.,   1.,   1.,
    #       1.,   1.,   8.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   7.,
    #       7.,   7.,   7.,   7.,   7.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
    #       1.,   3.,   3.,  14.,   1.,   1.,   1.,   1.,   3.,   2.,   1.,   7.,
    #       7.,   7.,   7.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
    #       1.,   1.,   1.,   1.,   1.,   7.,   7.,   7.,   7.,   7.,   7.,   1.]):
    #     logger.error("mapping: %s", mapping)
    #     logger.error("prob: %s", prob.config_str())
    #     logger.error("hw_config (normed): %s", hw_config)
    #     logger.error("hw_config (denormed): %s", hw_config_denormed)

    for i in range(len(hw_config)):
        s[f"dse.min_hw_{i}"] = float(hw_config[i])
    return s
def parallel_apply(chunk, arch_name, output_dir, prob_keys, dataset):
    return chunk.swifter.apply(add_col, args=(arch_name, output_dir, prob_keys, dataset), axis=1)

def add_min_hw_col(arch_name, output_dir, dataset):
    """
    make sure it works with all normalization
    """
    df: pd.DataFrame = dataset.df
    prob_keys = utils.keys_by_type(df, "prob")
    num_processes = 12
    pool = multiprocessing.Pool(processes=num_processes)
    chunks = np.array_split(dataset.df, num_processes)
    funcs = []
    for chunk in chunks:
        f = pool.apply_async(parallel_apply, (chunk,arch_name,output_dir,prob_keys,dataset))
        funcs.append(f)
    full_dataset = pd.DataFrame([])
    for i, f in enumerate(funcs):
        full_dataset = pd.concat([full_dataset, f.get()], ignore_index=True)
        funcs[i] = None
    # full_dataset = parallel_apply(dataset.df, arch_name, output_dir, prob_keys, dataset)
    dataset.df = full_dataset
    # pool.close()
    # df.swifter.apply(add_col, axis=1)

def search_network(arch_name: str, output_dir: pathlib.Path, workload: str, dataset_path: str, predictor: str="analytical", plot_only: bool=False, ordering: str="shuffle"):
    output_dir = pathlib.Path(output_dir).resolve()
    dataset_path = pathlib.Path(dataset_path).resolve()
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_bigarch_1_3_23")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_biggerarch_1_4_23_cosainit")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_biggerarch_1_4_23_resnet50")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_1_6_23")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_2_9_23")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_2_15_23")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_2_21_23")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_2_28_23")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_3_2_23")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_3_2_23_maxnorm")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_3_8_23_softplus")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_13_23_bw")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_13_23_bw_withmodel")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_25_23_nomodel")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_16_23_gemmini")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_16_23_gemmini_new")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_16_23_gemmini_allfeats")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_4_16_23_gemmini_rooflinemax")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_resnext_deepbench_alexnet_mm_3_2_23_maxnorm")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_allnets_3_2_23_logmean")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_debug")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_model_only")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_model_only_2")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_artifact")
#     # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_artifact_3")
#     output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_artifact_6")
#     workload = "resnet50"
    target = "edp"
    # gpu_id = random.choice([0, 1, 2, 3])
    gpu_id = None

    _DATA_DIR = DATASET_ROOT_PATH.parent.resolve() / "data"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_1000arch_1000map_12_6_22" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_bigarch_100000map_12_28_22" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_bigarch_100000map_1_3_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_biggerarch_100000map_1_4_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_defaultarch_cosamap_12_6_22" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_allnets_biggerarch_50000map_1_6_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_conv_test_100000arch_1map_12_29_22" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_allnets_1000arch_100map_2_9_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_allnets_biggerarch_2000map_2_23_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_allnets_biggerarch_1000map_3_29_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_allnets_dummyarch_100map_4_13_23" / "dataset.csv"
    # dataset_path = _DATA_DIR / "gemmini_allnets_100arch_10map_4_16_23" / "dataset.csv"
    # dataset_path = _DATA_DIR / "firesim_results.csv"
    # dataset_path = _DATA_DIR / "firesim_7_7_23" / "firesim_results.csv"
    # dataset_path = _DATA_DIR / "firesim_training_data" / "firesim_results.csv"
    # # dataset_path = _DATA_DIR / "firesim_6_14_23" / "dataset_firesim_6_14_23.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnet50_biggerarch_1000map_3_3_23" / "dataset.csv"
    # # dataset_path = _DATA_DIR / "gemmini_resnext_deepbench_alexnet_mm_biggerarch_1000map_3_3_23" / "dataset.csv"

    # sgd_lr = (1e-6 / 1.2**len(self.layers))
    # sgd_lr = 1e-8
    if target == "edp":
        if workload == "unet" or workload == "rnnt":
            sgd_lr = 5e-5
        elif workload == "resnet50" or workload == "retinanet":
            sgd_lr = 7.5e-5
        else:
            sgd_lr = 1e-5
        # sgd_lr = 3e-5
        # sgd_lr = 1e-4
    elif target == "energy":
        sgd_lr = 1e-4

    # if gumbel_perms:
    #     sgd_lr = sgd_lr * 2
    # if gumbel_perms == True:
    #     sgd_lr = sgd_lr * 8
    # else:
    #     sgd_lr = sgd_lr * 2
    # sgd_lr = sgd_lr * 8

    if "timeloop" in str(dataset_path):
        split_ratios = {"train": 0.001, "test": 0.999}
        constant_pe = False
    else:
        split_ratios = {"train": 0.92, "test": 0.08}
        constant_pe = True
    
    dataset_kwargs = {
        "dataset_path":dataset_path, "shuffle":True, "total_samples":10_000, "split_ratios":split_ratios, "process_mappings":"split",
        "target_log":False, "target_norm":"max", "probfeat_log":False, "probfeat_norm":"max",
        "archfeat_log":False, "archfeat_norm":"max", "mapfeat_log":False, "mapfeat_norm":"max", "num_processes":32,
    }
    # dataset_kwargs = {
    #     "dataset_path":dataset_path, "shuffle":True, "total_samples":50_000, "split_ratios":{"train": 0.95, "test": 0.05}, "process_mappings":"split",
    #     "target_log":False, "target_norm":"", "probfeat_log":False, "probfeat_norm":"",
    #     "archfeat_log":False, "archfeat_norm":"", "mapfeat_log":False, "mapfeat_norm":"", "num_processes":32,
    # }
    search_kwargs = {
        "sgd_lr": sgd_lr,
        "workload": workload,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "num_starts": 7,
        "base_gd_iters": 1490,
        "per_start_gd_step": 0,
        "round_steps": 500,
        "ordering_search_method": ordering,
        "gumbel_temp": 1,
        "shuffle_perms": True,
        "predictor": predictor,
        "plot_only": plot_only,
        "constant_pe": constant_pe,
    }
    config={
        "logfile": logger.parent.handlers[0].baseFilename,
        "search_kwargs": search_kwargs,
        "dataset_kwargs": dataset_kwargs,
    }
    logger.debug("config: %s", config)

    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="co-opt",
        
    #     # track hyperparameters and run metadata
    #     config=config,
    # )

    # seeds = [0, 100, 1111, 8888, 99999]
    seeds = [0]
    logger.debug("seeds: %s", seeds)
    for seed in seeds:
        network_searcher = MappingDrivenNetworkSearcher(arch_name, output_dir, workload, target, gpu_id=gpu_id)
        # bo_mappings, bo_results = network_searcher.search_bo(n_calls=100, n_initial_points=1, iters_per_layer=10)
        # hw_config, bo_mappings, bo_results = network_searcher.search_bo_v2(n_calls=100, n_initial_points=1, iters_per_layer=10)

        # dla_dataset_creator = DlaDatasetCreator(dataset_path=dataset_path, shuffle=True, total_samples=2_000_000, split_ratios={"train": 1}, process_mappings="split",
        #         target_log=True, target_norm="mean", probfeat_log=False, probfeat_norm="mean",
        #         archfeat_log=True, archfeat_norm="mean", mapfeat_log=False, mapfeat_norm="mean", num_processes=32)
        dla_dataset_creator = DlaDatasetCreator(**dataset_kwargs)
        train_data = dla_dataset_creator.get_train_data()

        gemmini_data_kwargs = dataset_kwargs.copy()
        # gemmini_test_df = pd.read_csv(DATASET_ROOT_PATH.parent / "data" / "firesim_test_data" / f"{predictor}.csv")
        gemmini_test_df = pd.read_csv(DATASET_ROOT_PATH.parent / "data" / "firesim_test_data" / f"dataset.csv")
        gemmini_test_df = gemmini_test_df.drop(columns=["target.gemmini_auto_cycle"])
        gemmini_test_df["target.cycle"] = gemmini_test_df["target.gemmini_cycle"]
        gemmini_data_kwargs["dataset_path"] = None
        gemmini_data_kwargs["df"] = gemmini_test_df
        gemmini_data_kwargs["split_ratios"] = {"train": 1}
        gemmini_data_kwargs["stats"] = dla_dataset_creator.stats
        gemmini_test_data = DlaDatasetCreator(**gemmini_data_kwargs).get_train_data()
        search_kwargs["extra_test_data"] = gemmini_test_data

        utils.set_random_seed(seed)
        if constant_pe:
            random_arch_configs = [init_hw_config(arch_name, [16, 2**random.randrange(0, 12, 1), 2**random.randrange(0, 12, 1)], output_dir) for _ in range(30)]
        else:
            random_arch_configs = [init_hw_config(arch_name, "random", output_dir) for _ in range(30)]
        # random_arch_configs = [init_hw_config(arch_name, [16, 128, 32], output_dir) for _ in range(search_kwargs["num_starts"])]

        # start_time = time.time()

        # # TODO: delete random generation code
        # random_arch_configs = [init_hw_config(arch_name, [16, 128, 32], output_dir)]
        # random_arch_configs = [init_hw_config(arch_name, [ 96, 657, 993], output_dir)]
        # # random_hw_random_results = [network_searcher.random_baseline(config.hw_config, 100000) for config in random_arch_configs]
        # random_hw_random_results = [network_searcher.random_baseline(config.hw_config, 1000) for config in random_arch_configs]
        # logger.info("random_hw_random_results: %s", random_hw_random_results)
        # return

        # runtime = time.time() - start_time
        # logger.error("Random runtime: %s", runtime)
        # exit(0)

        min_idx, hw_config_lst, best_mapping, bo_results, losses_lst, start_arch_configs_used, dfs, start_perf = network_searcher.search_gd(dla_dataset_creator, random_arch_configs, **search_kwargs)
        with open(output_dir / f"gd_best_result_{workload}.txt", "w") as f:
            f.write(str(min(bo_results)))

        same_hw_cosa_results = [network_searcher.cosa_baseline(hw_config_lst[i]) for i in range(len(hw_config_lst))]
        same_hw_random_results = [network_searcher.random_baseline(hw_config_lst[i], 1000) for i in range(len(hw_config_lst))]

        random_hw_random_results = [network_searcher.random_baseline(config.hw_config, 1000) for config in random_arch_configs[:10]]
        # default_hw_random_result = network_searcher.random_baseline([16, 128, 32], 1000)
        # random_hw_cosa_results = [network_searcher.cosa_baseline(config.hw_config) for config in random_arch_configs[:10]]
        # default_hw_cosa_result = network_searcher.cosa_baseline([16, 128, 32])

        hw_trial_size = 10
        sw_trial_size = 10
        sw_test_size = 1000
        hw_training_size = 100
        sw_training_size = 100
        probs, layers_counts = bo.get_layers(workload)
        training_configs, training_val, best_config, best_edp = bo.hw_optimize(probs, layers_counts, hw_training_size, sw_training_size, hw_trial_size, sw_trial_size, sw_test_size)
        actual_bo_results = list(training_val) + [best_edp]

        # logger.info("Searched mappings: %s", best_mapping)
        # logger.info("Searched HW: %s", hw_config_lst[min_idx])
        # logger.info("Searched results predicted: [%s]", ', '.join(['{:.3e}'.format(x) for x in losses_lst]))
        # logger.info("Searched results predicted best: %s", min(losses_lst))
        # logger.info("Searched results real: [%s]", ', '.join(['{:.3e}'.format(x) for x in bo_results]))
        # logger.info("Searched results real best: %s", min(bo_results))
        # logger.info("Start perf: %s", start_perf)
        # logger.info("Same HW CoSA results: [%s]", ', '.join(['{:.3e}'.format(x) for x in same_hw_cosa_results]))
        # logger.info("Same HW CoSA best result: %s", min(same_hw_cosa_results))
        # logger.info("Same HW random mapper results: [%s]", ', '.join(['{:.3e}'.format(x) for x in same_hw_random_results]))
        # logger.info("Same HW random mapper best result: %s", min(same_hw_random_results))
        # logger.info("Default HW CoSA result: %s", default_hw_cosa_result)
        # logger.info("Random HW CoSA results: [%s]", ', '.join(['{:.3e}'.format(x) for x in random_hw_cosa_results]))
        # logger.info("Random HW CoSA best result: %s", min(random_hw_cosa_results))
        # logger.info("Default HW random mapper result: %s", default_hw_random_result)
        # logger.info("Random HW random mapper results: [%s]", ', '.join(['{:.3e}'.format(x) for x in random_hw_random_results]))
        # logger.info("Random HW random mapper best result: %s", min(random_hw_random_results))

        if target == "edp":
            ylabel = "Energy-delay product (uJ * Cycles)"
        elif target == "adp":
            ylabel = "Area-delay product (mm^2 * Cycles)"
        elif target == "energy":
            ylabel = "Energy (uJ)"
        
        # plot_width = max(len(bo_results), len(random_hw_cosa_results), len(random_hw_random_results))
        # # bo_results = [5235116.97408, 850826.107896, 831384.5119800001, 846905.287136, 902580.165332, 847203.330856, 867371.583224, 2276861.44, 1537811.00294, 4488557.101056]
        # # same_hw_cosa_result = 953450.7397800001
        # # default_hw_cosa_result = 4465153.500539999
        # # random_hw_cosa_results = [10118517.248, 24912355.39968, 6412851.500376, 72473637.376, 42896315.392000005, 6093909.230892, 6587545.927482, 16966204.157952, 22174254.592, 3276011.445291]
        # plt.figure(dpi=300)
        # plt.plot(range(1,1+len(bo_results)), utils.search_curve(bo_results), label="mapping-first search", color="tab:blue")
        # plt.plot(range(1,1+len(bo_results)), utils.search_curve(same_hw_cosa_results), label="mapping-first HW, CoSA mapping", linestyle="dashed", color="tab:blue")
        # plt.plot(range(1,1+len(bo_results)), utils.search_curve(same_hw_random_results), label="mapping-first HW, 1000 random mappings", linestyle="dotted", color="tab:blue")
        # plt.plot(range(1,1+plot_width), [default_hw_cosa_result] * plot_width, label="default HW, CoSA mapping", linestyle="dashed", color="tab:red")
        # plt.plot(range(1,1+plot_width), [default_hw_random_result] * plot_width, label="default HW, 1000 random mappings", linestyle="dotted", color="tab:red")
        # plt.plot(range(1,1+len(random_hw_cosa_results)), utils.search_curve(random_hw_cosa_results), label="random HW, CoSA mapping", linestyle="dashed", color="tab:green")
        # plt.plot(range(1,1+len(random_hw_random_results)), utils.search_curve(random_hw_random_results), label="random HW, 1000 random mappings", linestyle="dotted", color="tab:green")
        # plt.title(workload)
        # plt.legend()
        # plt.xlabel("Evaluated HW Configs")
        # plt.ylabel(ylabel)
        # # plt.savefig(output_dir / utils.unique_filename("png", f"network_searcher_{workload}"), bbox_inches="tight")
        # # plt.savefig(output_dir / utils.unique_filename("pdf", f"network_searcher_{workload}"), bbox_inches="tight")

        plt.figure(dpi=300)
        plt.plot([1] + list(np.arange(1,len(bo_results)+1) * 1500), utils.search_curve([start_perf] + list(bo_results)), label="dosa", color="tab:blue")
        # plt.plot(np.arange(1,len(bo_results)+1) * 1500, utils.search_curve(same_hw_cosa_results), label="mapping-first HW, CoSA mapping", linestyle="dashed", color="tab:blue")
        # plt.plot(range(1,1+len(bo_results)), utils.search_curve(same_hw_random_results), label="mapping-first HW, 1000 random mappings", linestyle="dotted", color="tab:blue")
        # plt.plot(range(1,1+len(bo_results)), [default_hw_cosa_result] * len(bo_results), label="default HW, CoSA mapping", linestyle="dashed", color="tab:red")
        # plt.plot(range(1,1+len(bo_results)), [default_hw_random_result] * len(bo_results), label="default HW, 1000 random mappings", linestyle="dotted", color="tab:red")
        # plt.plot(range(1,1+len(random_hw_cosa_results)), utils.search_curve(random_hw_cosa_results), label="random HW, CoSA mapping", linestyle="dashed", color="tab:green")
        plt.plot(np.arange(1,len(random_hw_random_results)+1)*1000, utils.search_curve(random_hw_random_results), label="random", color="tab:orange")
        plt.plot(list(np.arange(1,len(actual_bo_results))*100) + [(len(actual_bo_results)-1)*100+1], utils.search_curve(actual_bo_results), label="bo", color="tab:green")
        plt.yscale("log")
        plt.title(workload)
        plt.legend()
        plt.xlabel("Timeloop/Differentiable Model Samples")
        plt.ylabel(ylabel + " (log-scale)")
        plt.savefig(output_dir / utils.unique_filename("png", f"network_searcher_{workload}_log"), bbox_inches="tight")
        # plt.savefig(output_dir / utils.unique_filename("pdf", f"network_searcher_{workload}_log"), bbox_inches="tight")

def get_hw_bounds(arch_name: str) -> list[tuple]:
    bounds = []
    if arch_name == "gemmini":
        bounds = [
            skopt.space.Integer(1, GemminiConfig.BASE_PE * 8, name="pe_dim"),
            skopt.space.Integer(1, GemminiConfig.BASE_SP_SIZE * 8, name="sp_size"),
            skopt.space.Integer(1, GemminiConfig.BASE_ACC_SIZE * 8),
        ]
    return bounds

def get_mapping_bounds(arch_name: str, prob: Prob) -> list[tuple]:
    num_mem_lvls = utils.num_mem_lvls(arch_name)
    
    dim_idx_dict = prob.prob_name_idx_dict
    num_dims = len(dim_idx_dict)
    bounds = [tuple()[:]] * num_mem_lvls * num_dims * 3
    for mem_lvl in range(num_mem_lvls):
        start_idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3
        end_idx = start_idx + num_dims * 3
        mem_lvl_bounds = bounds[start_idx:end_idx]
        # spatial factors
        for dim, idx in dim_idx_dict.items():
            total_idx = start_idx + idx
            name = f"memlvl{mem_lvl}_dim{dim}_spatial"
            # bounds[total_idx] = Categorical(prob.prob_divisors[idx])
            spatial_cond = False
            if arch_name == "gemmini": # only parallelize C and K dims
                spatial_cond = spatial_cond or (mem_lvl == 1 and dim == "C")
                spatial_cond = spatial_cond or (mem_lvl == 2 and dim == "K")
            spatial_cond = spatial_cond and prob.prob[dim] > 1
            if spatial_cond:
                bound = skopt.space.Integer(1, prob.prob[dim], name=name)
            else:
                bound = skopt.space.Categorical((1,), name=name)
            bounds[total_idx] = bound

        # temporal factors
        for dim, idx in dim_idx_dict.items():
            total_idx = start_idx + num_dims + idx
            name = f"memlvl{mem_lvl}_dim{dim}_temporal"
            temporal_cond = False
            if arch_name == "gemmini":
                temporal_cond = not (mem_lvl == 0 and dim not in ("P", "Q",)) # to accommodate cosa mappings
            temporal_cond = temporal_cond and prob.prob[dim] > 1
            if temporal_cond:
                bound = skopt.space.Integer(1, prob.prob[dim], name=name)
            else:
                bound = skopt.space.Categorical((1,), name=name)
            bounds[total_idx] = bound

        # perms
        for dim, idx in dim_idx_dict.items():
            total_idx = start_idx + 2 * num_dims + idx
            bounds[total_idx] = skopt.space.Real(0, 100, name=f"memlvl{mem_lvl}_dim{dim}_perm")

    return bounds

def random_search_space(arch_name: str, output_dir: pathlib.Path, prob: Prob, space: skopt.Space, n_samples: int):
    samples = space.rvs(n_samples=n_samples)
    result_inputs = []
    result_targets = []
    for orig_mapping in samples:
        mapping = mapping_utils.round_mapping(orig_mapping, prob)
        hw_config, cap_per_layer, max_cap_idxs = min_hw(arch_name, output_dir, [(mapping, prob)])
        arch_config = init_hw_config(arch_name, hw_config, output_dir)
        row = arch_config.run_mapping_from_dict(prob, arch_config.flat_mapping_to_dict(prob.shape, mapping))
        print(row)
        try:
            target = row["target.cycle"] * row["target.energy"]
        except Exception:
            logger.error("Could not run min_hw for rounded mapping %s on arch %s, prob %s\n%s",
                           mapping, arch_name, prob.config_str(), traceback.format_exc())
            target = np.finfo("float64").max

        result_inputs.append(orig_mapping)
        result_targets.append(target)
    return result_inputs, result_targets

def bo_search_hw(arch_name, output_dir, prob, n_calls, n_initial_points) -> tuple[list, list]:
    def minimize_hw_fn(hw_config):
        arch_config = init_hw_config(arch_name, hw_config, output_dir)
        rows = arch_config.run_cosa(prob)
        print(rows)
        try:
            row = rows[0]
            return row["target.cycle"] * row["target.energy"]
        except Exception:
            return np.finfo("float64").max

    bounds = get_hw_bounds(arch_name)
    res_gp = skopt.gp_minimize(minimize_hw_fn, 
                               bounds, 
                               n_calls=n_calls,
                               n_initial_points=n_initial_points,
                               n_jobs=16,
                               random_state=0)
    return res_gp.x_iters, res_gp.func_vals

def search_layer(arch_name: str, output_dir: pathlib.Path, prob: Prob, n_calls: int, n_initial_points: int) -> dict:
    def minimize_mapping_fn(mapping):
        mapping = mapping_utils.round_mapping(mapping, prob)
        hw_config, cap_per_layer, max_cap_idxs = min_hw(arch_name, output_dir, [(mapping, prob)])
        arch_config = init_hw_config(arch_name, hw_config, output_dir)
        row = arch_config.run_mapping_from_dict(prob, arch_config.flat_mapping_to_dict(prob.shape, mapping))
        print(row)
        try:
            return row["target.cycle"] * row["target.energy"]
        except Exception:
            logger.error("Could not run min_hw for rounded mapping %s on arch %s, prob %s\n%s",
                            mapping, arch_name, prob.config_str(), traceback.format_exc())
            return np.finfo("float64").max

    bounds = get_mapping_bounds(arch_name, prob)

    gen = CosaPointGenerator(arch_name, output_dir, prob)
    # initial_point_generator = skopt.utils.cook_initial_point_generator(gen)
    res_gp = skopt.gp_minimize(minimize_mapping_fn, 
                               bounds, 
                               n_calls=n_calls,
                               n_initial_points=0,
                               n_jobs=16,
                            #    initial_point_generator=initial_point_generator,
                               x0=gen.generate(None, n_initial_points, None),
                               random_state=0)
    gp_targets = res_gp.func_vals
    space = skopt.Space(bounds)
    random_inputs, random_targets = random_search_space(arch_name, output_dir, prob, space, n_samples=n_calls)
    random_hw_inputs, random_hw_targets = gen._generate_results(bounds, n_calls)
    bo_hw_inputs, bo_hw_targets = bo_search_hw(arch_name, output_dir, prob, n_calls, n_initial_points)

    plot_after = 5
    plt.figure()
    plt.plot(range(1, len(gp_targets)+1), utils.search_curve(gp_targets), label="skopt.gp_minimize mapping-first")
    # plt.plot(range(1, len(random_targets)+1)[plot_after:], utils.search_curve(random_targets)[plot_after:], label="random mappings -> HW")
    plt.plot(range(1, len(random_hw_targets)+1)[plot_after:], utils.search_curve(random_hw_targets)[plot_after:], label="random HW + CoSA mapping")
    plt.plot(range(1, len(bo_hw_targets)+1), utils.search_curve(bo_hw_targets), label="BO HW + CoSA mapping")
    plt.xlabel("Iteration")
    plt.ylabel("EDP (pJ * MCycles)")
    plt.title(prob.config_str())
    plt.legend()
    plt.savefig(output_dir / utils.unique_filename("png", f"gp_convergence_{n_initial_points}points_{n_calls}calls"), bbox_inches="tight")

    plot_after = len(random_targets) // 2
    plt.figure()
    plt.plot(range(1, len(gp_targets)+1)[plot_after:], utils.search_curve(gp_targets)[plot_after:], label="skopt.gp_minimize")
    # plt.plot(range(1, len(random_targets)+1)[plot_after:], utils.search_curve(random_targets)[plot_after:], label="random mappings -> HW")
    plt.plot(range(1, len(random_hw_targets)+1)[plot_after:], utils.search_curve(random_hw_targets)[plot_after:], label="random HW + CoSA")
    plt.plot(range(1, len(bo_hw_targets)+1)[plot_after:], utils.search_curve(bo_hw_targets)[plot_after:], label="BO HW + CoSA mapping")
    # fig.axes[0].set_xlim(left=plot_after)
    plt.xlabel("Iteration")
    plt.ylabel("EDP (pJ * MCycles)")
    plt.title(prob.config_str())
    plt.legend()
    plt.savefig(output_dir / utils.unique_filename("png", f"gp_convergence_{n_initial_points}points_{n_calls}calls_second_half"), bbox_inches="tight")

    return res_gp

def min_hw(arch_name: str, output_dir: pathlib.Path, mapping_prob_pairs: list[tuple[list[int], Prob]], grad=False) -> tuple[list[dict], list[float], list[int]]:
    """Computes minimum size hw needed to run provided mappings
    """
    cap_per_layer = []
    if arch_name == "gemmini":
        pe_dim_max_idx = 0
        sp_cap_max_idx = 0
        acc_cap_max_idx = 0
        pe_dim_max = 0
        sp_cap_max = 0
        acc_cap_max = 0

        max_per_layer = None

        for idx, (mapping, prob) in enumerate(mapping_prob_pairs):
            mac_needed, max_spatial_factor, buf_needed = mapping_utils.capacity_from_mapping(mapping, prob)
            # logger.info("%s %s %s", mac_needed, max_spatial_factor, buf_needed)
            # pe_dim = math.ceil(mac_needed ** 0.5)
            sp_cap = buf_needed[2][0] + buf_needed[2][1] # W + I
            spatial_k_idx = len(prob.prob_name_idx_dict) * 3 + 5
            spatial_k = mapping[spatial_k_idx]
            acc_cap = buf_needed[1][2]# * spatial_k # O
            # logger.info("%s %s %s", max_spatial_factor, sp_cap, acc_cap)
            this_layer_max = torch.cat((max_spatial_factor.unsqueeze(-1), sp_cap.unsqueeze(-1), acc_cap.unsqueeze(-1)))
            if max_per_layer is None:
                max_per_layer = this_layer_max.unsqueeze(0)
            else:
                max_per_layer = torch.cat((max_per_layer, this_layer_max.unsqueeze(0)), dim=0)
            # logger.debug("%s %s %s, spatial_k %s", pe_dim, sp_cap, acc_cap, spatial_k)
            cap_per_layer.append([max_spatial_factor, sp_cap, acc_cap])
            if max_spatial_factor > pe_dim_max:
                pe_dim_max = max_spatial_factor
                pe_dim_max_idx = idx
            if sp_cap > sp_cap_max:
                sp_cap_max = sp_cap
                sp_cap_max_idx = idx
            if acc_cap > acc_cap_max:
                acc_cap_max = acc_cap
                acc_cap_max_idx = idx

        vals, idxs = torch.max(max_per_layer, dim=0)

        # Construct Gemmini instance
        # PE dim 1 doesn't work
        # pe_dim_max = max(2, pe_dim_max)
        pe_dim_max = torch.clamp(vals[0], min=2)

        # sp specified in KB
        if grad:
            sp_cap_max = vals[1] / 1024
        else:
            sp_cap_max = math.ceil(sp_cap_max / 1024)

        # acc uses 4B words; hw_config specified in KB
        # multiply by pe_dim_max to handle spatial dim being moved to sp
        if grad:
            acc_cap_max = pe_dim_max * vals[2] * 4 / 1024
        else:
            acc_cap_max = math.ceil(pe_dim_max * acc_cap_max * 4 / 1024)

        if grad:
            hw_config = torch.cat((pe_dim_max.unsqueeze(-1), sp_cap_max.unsqueeze(-1), acc_cap_max.unsqueeze(-1)))
        else:
            hw_config = [int(pe_dim_max), float(sp_cap_max), float(acc_cap_max)]

        max_cap_idxs = [pe_dim_max_idx, sp_cap_max_idx, acc_cap_max_idx]
    # logger.debug("min hw_config: %s", hw_config)
    return hw_config, cap_per_layer, max_cap_idxs

if __name__ == "__main__":
    output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_debug")
    # output_dir = pathlib.Path("output_dir_mapping_driven_hw_search_groupmeeting")
    prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv" / "conv_0.yaml")
    n_calls = 100
    n_initial_points = 5

    # mapping = [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  7.,  7.,  1.,  1.,
    #     1.,  4.,  5.,  1.,  2.,  3.,  6.,  7.,  1.,  1.,  1.,  1.,  1.,
    #     8.,  1.,  1.,  7.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  3.,  4.,
    #     5.,  6.,  7.,  1.,  1.,  1.,  1.,  3.,  1.,  1.,  7.,  1.,  1.,
    #     4.,  1.,  8.,  1.,  2.,  4.,  5.,  1.,  6.,  3.,  7.,  1.,  1.,
    #     1.,  1.,  1.,  1.,  1.,  1.,  1., 16.,  4.,  1.,  1.,  1.,  3.,
    #     4.,  1.,  2.,  5.,  6.,  7.]
    # print(min_hw("gemmini", output_dir, [(mapping, prob)]))
    # exit(0)
    # i = 2
    # prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv" / f"conv_{i}.yaml")
    # res_gp = search_layer("gemmini", output_dir, prob, n_calls=n_calls, n_initial_points=n_initial_points)
    # logger.info("Best mapping: %s, best result: %s", res_gp.x, res_gp.fun)
    # logger.debug("%s", res_gp)

    _DATA_DIR = DATASET_ROOT_PATH.parent.resolve() / "data"
    dataset_path = _DATA_DIR / "firesim_training_data" / "firesim_results.csv"
    search_network("gemmini", output_dir, "conv", dataset_path)

    # prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv" / "conv_0.yaml")
    # arch_config = GemminiConfig([16, 256, 64], output_dir)
    # row = arch_config.run_cosa(prob)[0]
    # print(row["target.cycle"] * row["target.energy"])

    # arch_config = GemminiConfig([64, 5, 4], output_dir)
    # # print(arch_config.run_cosa(prob))
    # rows = arch_config.run_random_mappings(prob, 10000)
    # min_row = None
    # min_edp = float("inf")
    # for row in rows:
    #     if row["target.edp"] < min_edp:
    #         min_row = row
    #         min_edp = row["target.edp"]
    # print(min_row)
    
    # conv_mappings = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 2.0, 1.0, 1.0, 1.0, 6.0, 6.0, 0.0, 1.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 7.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 0.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 28.0, 28.0, 1.0, 1.0, 1.0, 6.0, 6.0, 0.0, 1.0, 6.0, 6.0, 6.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 0.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 1.0, 1.0, 1.0, 28.0, 4.0, 1.0, 1.0, 1.0, 6.0, 6.0, 0.0, 1.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 14.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 0.0, 6.0, 6.0, 6.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 1.0, 1.0, 3.0, 14.0, 14.0, 2.0, 4.0, 1.0, 6.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 64.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0, 2.0, 8.0, 1.0, 6.0, 6.0, 0.0, 6.0, 1.0, 2.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 0.0, 6.0, 6.0, 6.0]]

    # mapping_prob_pairs = []
    # for i, mapping in enumerate(conv_mappings):
    #     prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv" / f"conv_{i}.yaml")
    #     mapping = mapping_utils.round_mapping(mapping, prob)
    #     mapping_prob_pairs.append((mapping, prob,))        

    # # orig_mapping = [1, 1, 1, 1, 1, 1, 1, 2, 3, 14, 14, 201, 256, 1, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 256, 1, 1, 3, 1, 1, 1, 256, 1, 0.0, 77.54897885998903, 100.0, 100.0, 100.0, 100.0, 100.0, 1, 1, 1, 1, 256, 1, 1, 1, 1, 7, 14, 1, 1, 1, 100.0, 100.0, 0.0, 100.0, 0.0, 51.007377514578934, 34.911502569780126, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 0.0, 100.0, 21.133347976235914, 0.0, 86.52999056020555, 100.0, 100.0]
    # # mapping = mapping_utils.round_mapping(orig_mapping, prob)
    # # print(mapping)
    # # hw_config, _, _ = min_hw("gemmini", output_dir, mapping_prob_pairs)
    # hw_config = [16, 256, 64]
    # print(hw_config)
    # arch_config = init_hw_config("gemmini", hw_config, output_dir)
    # rows = []
    # for mapping, prob in mapping_prob_pairs:
    #     # row = arch_config.run_mapping_from_dict(prob, arch_config.flat_mapping_to_dict(prob.shape, mapping))
    #     row = arch_config.run_cosa(prob)[0]
    #     rows.append(row)
    # print(rows)
    # print(sum([row["target.cycle"] * row["target.energy"] for row in rows]))

    # example_mapping = mapping_utils.process_mapping("L3[WIO] C8 Q7 K16 - L2[WI] Q2 C2 - L1[O] P7 K2 C4 K2X - L0[W] N1", prob.shape)
    # example_mapping_2 = mapping_utils.process_mapping("L3[WIO] C8 Q7 K4 - L2[WI] Q2 C2 P2X C32X - L1[O] P7 K2 C4 K4X - L0[W] N1", prob.shape)
    # mapping_prob_pairs = [
    #     (example_mapping, prob),
    #     (example_mapping_2, prob),
    # ]
    # example_mapping_3 = mapping_utils.process_mapping("L3[WIO] C8 Q7 K4 - L2[WI] Q2 C2 - L1[O] P7 K200 C4 K8X - L0[W] N1", prob.shape)
    # example_mapping_3 = mapping_utils.round_mapping(example_mapping_3, prob)
    # print(example_mapping_3)
    # mapping_prob_pairs.append((example_mapping_3, prob))
    # rows, _, _ = min_hw("gemmini", output_dir, mapping_prob_pairs)
    # print(rows)
