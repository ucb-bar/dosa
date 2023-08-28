#!/usr/bin/env python3 
import argparse
import pathlib
import csv
import itertools
import traceback

from dataset import DATASET_ROOT_PATH
from dataset.common import utils, logger
from dataset.hw import HardwareConfig, GemminiConfig, SimbaConfig
from dataset.workloads import Prob

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument(
                        '--layer_idx',
                        type=str,
                        help='Target DNN Layer',
                        default='',
                        )
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_random',
                        )
    parser.add_argument('-an',
                        '--arch_name',
                        type=str,
                        help='Hardware Architecture Name [gemmini, simba]',
                        default='gemmini',
                        )
    parser.add_argument(
                        '--arch_file',
                        type=str,
                        help='Optional: path to a single arch YAML file to use',
                        default=None,
                        )
    parser.add_argument('-bwp',
                        '--base_workload_path',
                        type=str,
                        help='Base Workload Path',
                        default=f'{DATASET_ROOT_PATH}/workloads/'
                        )
    parser.add_argument('-wl',
                        '--workload',
                        action='append',
                        help='<Required> Name of workload directory. Use flag \
                              multiple times for multiple workloads, e.g. \
                              `-wl mm -wl conv` ...',
                        required=True,
                        )
    parser.add_argument(
                        '--random_seed',
                        type=int,
                        help='Random Seed',
                        default=1,
                        )
    parser.add_argument(
                        '--num_arch',
                        type=int,
                        help='Number of random arch to run',
                        default=1,
    )
    parser.add_argument(
                        "--mapper",
                        type=str,
                        help="which mapper to use [random, cosa]",
                        default="random",
                        )
    parser.add_argument(
                        '--num_mappings',
                        type=int,
                        help='Number of mappings per problem/hardware config',
                        default=1000,
                        )
    parser.add_argument(
                        '--exist',
                        action='store_true',
                        help='Set flag if data already exists but needs to be compiled to csv',
                        )
    parser.add_argument(
                        '--min_metric',
                        type=str,
                        help='Save only the minimum mapping according to this metric. If not specified,\
                              save all mappings.',
                        default=None,
                        )
    return parser


def run_gemmini(layers, output_dir, num_mappings, exist):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    buf_multipliers = [x/2 for x in range(1, 9, 1)]
    buf_multipliers_perms = [p for p in itertools.product(buf_multipliers, repeat=2)]

    # buf_attributes = {
    #     1: {"depth": mem1_depth, "blocksize": mem1_blocksize, "ports": mem1_ports, "banks": mem1_banks}, # scratchpad
    #     2: {"depth": mem2_depth, "blocksize": mem2_blocksize, "ports": mem2_ports, "banks": mem2_banks}, # acc
    # }

    header_written = False
    header_keys = {}

    # Write to CSV
    for pe_multiplier in [0.5, 1, 2, 4]:
        for buf_multipliers_perm in buf_multipliers_perms:
            hw_config = [
                int(GemminiConfig.BASE_PE * pe_multiplier),
                int(GemminiConfig.BASE_SP_SIZE * buf_multipliers_perm[0]),
                int(GemminiConfig.BASE_ACC_SIZE * buf_multipliers_perm[1]),
            ]
            gemmini_config = GemminiConfig(hw_config, output_dir / "logs")

            for layer_path in layers:
                try:
                    rows = gemmini_config.run_random_mappings(layer_path, num_mappings, exist)
                except Exception:
                    traceback.print_exc()
                    continue

                # Write to CSV
                if not header_written:
                    with open(output_dir / "dataset.csv", "w") as f:
                        w = csv.DictWriter(f, rows[0].keys())
                        w.writeheader()
                        header_written = True
                        header_keys = rows[0].keys()
                else:
                    with open(output_dir / "dataset.csv", "a") as f:
                        w = csv.DictWriter(f, header_keys)
                        w.writerows(rows)

    # Write compressed version of csv
    utils.make_tarfile(output_dir / f"dataset.csv.tar.gz", output_dir / "dataset.csv")

def run(arch_name, arch_file, layers, output_dir, num_arch, mapper, num_mappings, exist, min_metric=None):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    header_written = False
    header_keys = {}

    # if arch_file set, override num_arch and only run 1
    if arch_file:
        num_arch = 1

    # Run random arch and write to CSV
    for arch_i in range(num_arch):
        if arch_file:
            hw_config = pathlib.Path(arch_file).resolve()
        else:
            hw_config = "random"

        if arch_name == "gemmini":
            arch_config = GemminiConfig(hw_config, output_dir / "logs")
        elif arch_name == "simba":
            arch_config = SimbaConfig(hw_config, output_dir / "logs")

        for layer_path in layers:
            layer_prob = Prob(layer_path)
            return_min_fn = None
            if min_metric:
                if min_metric == "cycle" or min_metric == "energy":
                    return_min_fn = lambda row: row[f"target.{min_metric}"]
                elif min_metric == "edp":
                    return_min_fn = lambda row: row["target.cycle"] * row["target.energy"]
            try:
                if mapper == "random":
                    rows = arch_config.run_random_mappings(layer_prob, num_mappings, exist, return_min_fn=return_min_fn)
                else:
                    rows = arch_config.run_cosa(layer_prob, exist)
            except Exception:
                traceback.print_exc()
                continue

            # Empty dict rows probably means Timeloop crashed on this run
            if not rows:
                logger.error("Generated no rows for arch %s, layer %s", arch_config.get_config_str(), layer_prob.config_str())
                continue

            # Write to CSV
            dataset_path: pathlib.Path = output_dir / "dataset.csv"
            if not header_written: # create dataset csv file
                with open(dataset_path, "w") as f:
                    w = csv.DictWriter(f, rows[0].keys())
                    w.writeheader()
                    header_written = True
                    header_keys = rows[0].keys()
                    w.writerows(rows)
            else: # add to existing file
                with open(dataset_path, "a") as f:
                    w = csv.DictWriter(f, header_keys)
                    w.writerows(rows)

        if arch_i == num_arch - 1 or arch_i % 100 == 0:
            logger.info("Ran %d of %d arch", arch_i+1, num_arch)

    # Write compressed version of csv
    utils.make_tarfile(output_dir / f"dataset.csv.tar.gz", output_dir / "dataset.csv")


if __name__ == "__main__":
    # test_config = GemminiConfig([128, 1024, 1024], "dummy_gemmini")
    # exit(0)
    parser = construct_argparser()
    args = parser.parse_args()

    logger.info("Running run.py with args %s", args)

    random_seed = args.random_seed

    if args.layer_idx:
        layer_idx = args.layer_idx
    else:
        layer_idx = None

    output_dir = args.output_dir
    output_dir = f'{output_dir}' 
    if layer_idx: 
        output_dir += f'_layer{layer_idx}'
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_workload_path = pathlib.Path(args.base_workload_path).resolve()
    # workloads = ['conv', 'mm']
    workloads = args.workload

    layers = []
    for workload in workloads:
        workload_path = base_workload_path / workload
        unique_layers = utils.parse_yaml(workload_path / 'unique_layers.yaml')
        for unique_layer in unique_layers:
            layer_path = workload_path / (unique_layer+'.yaml')
            layer_path = layer_path.resolve()
            layers.append(layer_path)

    valid_arch = ["gemmini", "simba"]
    if args.arch_name not in valid_arch:
        logger.error("Arch %s not implemented. Try one of %s", args.arch_name, valid_arch)
        exit(1)

    utils.set_random_seed(args.random_seed)
    # run_gemmini(layers, output_dir, num_mappings=args.num_mappings, exist=args.exist)
    run(args.arch_name, args.arch_file, layers, output_dir, num_arch=args.num_arch, mapper=args.mapper, 
        num_mappings=args.num_mappings, exist=args.exist, min_metric=args.min_metric)
