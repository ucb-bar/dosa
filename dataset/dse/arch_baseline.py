import pathlib
import os
import re
import subprocess
import time
import argparse

import matplotlib.pyplot as plt

from dataset import DATASET_ROOT_PATH
from dataset.common import utils, logger
from dataset.workloads import Prob

_TIMELOOP_MAPPER_PATH = "timeloop-mapper"
def run_timeloop(paths, cwd=os.getcwd(), stdout=None, stderr=None, run_async=False):
    args = [_TIMELOOP_MAPPER_PATH]
    for path in paths:
        args.append(pathlib.Path(path).resolve())
    logger.info('run_timeloop> {}'.format(args))
    try:
        if not run_async:
            p = subprocess.check_call(args, cwd=cwd, stdout=stdout, stderr=stderr)
            return True
        else:
            p = subprocess.Popen(args, cwd=cwd, stdout=stdout, stderr=stderr)
            return p
    except Exception as e:
        logger.error("Failed to run timeloop-mapper with exception %s", e)
        return False

def get_layer_count(workload_path, layers) -> list[int]:
    try:
        layer_count_dict = utils.parse_yaml(workload_path / 'layer_count.yaml')
        counts = [layer_count_dict[prob.config_str()]["count"] for prob in layers]
    except:
        logger.warning("Couldn't find layer count, using default layer counts")
        counts = [1 for prob in layers]
    return counts

def parse_random_output(output_log_file: pathlib.Path, layer_prob: Prob) -> list[dict]:
    """
    Parses stdout of Timeloop that spits out all randomly searched schedules

    Returns:
        List of dictionaries containing arch, prob, mapping, and target data
        for each mapping.
    """
    # Format prob and arch features for csv
    prob_dict_flattened = {"prob.shape": layer_prob.shape}
    # Handle lack of instance key (old format)
    # Might get shape twice, but should be ok
    prob_instance = layer_prob.prob
    for key, val in prob_instance.items():
        prob_dict_flattened[f"prob.{key}"] = val

    min_cycle = 1e100

    rows = list()

    try:
        with open(output_log_file, "r") as f:
            lines = f.readlines()
    except IOError as _:
        logger.error("Couldn't read file %s", output_log_file)
        return rows

    for line_num in range(len(lines)):
        # if lines[l] == "---------------------------\n":
        if re.match(r"=== Summary ===\n", lines[line_num]):
            row = dict()
            cycle = int(re.match(r"Cycles: ([0-9.e+-]+)", lines[line_num + 1]).group(1))
            energy = float(re.match(r"Energy: ([0-9.e+-]+)", lines[line_num + 2]).group(1))
            edp = float(re.match(r"EDP\(J\*cycle\): ([0-9.e+-]+)", lines[line_num + 3]).group(1))
            area = float(re.match(r"Area: ([0-9.e+-]+)", lines[line_num + 4]).group(1))
            row["target.cycle"] = cycle
            row["target.energy"] = energy
            row["target.edp"] = edp
            row["target.area"] = area

            mapping = lines[line_num + 6].strip()
            row["mapping.mapping"] = mapping

            # if cycle < min_cycle:
            #     min_cycle = cycle

            rows.append(row)
    return rows

def parse_timeloop_output(output_path, layer_prob, return_min_fn=None):
    output_log_file = output_path / "random.txt"
    rows = parse_random_output(output_log_file, layer_prob)
    if not rows:
        logger.warning("Failed to exhaustively run mappings, output at %s", output_path)
    else:
        logger.info("Ran %d mappings", len(rows))
    if return_min_fn:
        min_idx = -1
        min_val = float("inf")
        for idx, row in enumerate(rows):
            val = return_min_fn(row)
            if val < min_val:
                min_idx = idx
                min_val = val
        return [rows[min_idx]]

    return rows


def main(output_dir, workload):
    output_dir = pathlib.Path(output_dir).resolve()
    arch_name = "gemmini"

    base_workload_path = DATASET_ROOT_PATH / "workloads"
    workload_path = base_workload_path / workload
    workload_name = workload

    layers = []
    unique_layers = utils.parse_yaml(workload_path / 'unique_layers.yaml')
    for unique_layer in unique_layers:
        layer_path = workload_path / (unique_layer+'.yaml')
        layer_path = layer_path.resolve()
        layers.append(Prob(layer_path))
    layer_count = get_layer_count(workload_path, layers)
    logger.info("Layer count: %s", layer_count)

    # get result from prev fig5.sh run
    try:
        with open(output_dir / f"gd_best_result_{workload}.txt", "r") as f:
            line = f.readline()
            gd_result = float(line)
    except:
        logger.error("Could not find DOSA result for %s", workload)
        return

    results = {}
    for arch_name in ["eyeriss", "nvdla-small", "nvdla-large", "gemmini"]:
        cycle_total = 0
        energy_total = 0

        output_paths = []
        procs = []
        for layer_idx, prob in enumerate(layers):
            if arch_name == "eyeriss":
                paths = [ # Eyeriss
                    DATASET_ROOT_PATH.parent / "accelergy-timeloop-infrastructure" / "src" / "timeloop" / "configs" / "mapper" / "eyeriss-256-split-accelergy-arg.yaml",
                    DATASET_ROOT_PATH.parent / "accelergy-timeloop-infrastructure" / "src" / "timeloop" / "configs" / "mapper" / "eyeriss-256-split-problem.yaml",
                    prob.path
                ]
            elif "nvdla" in arch_name:
                if "small" in arch_name:
                    paths = [ # NVDLA small
                        DATASET_ROOT_PATH / "hw" / "nvdla" / "arch" / "arch_small.yaml",
                        DATASET_ROOT_PATH / "hw" / "nvdla" / "mapspace" / "mapspace.yaml",
                        prob.path
                    ]
                else:
                    paths = [ # NVDLA large
                        DATASET_ROOT_PATH / "hw" / "nvdla" / "arch" / "arch_large.yaml",
                        DATASET_ROOT_PATH / "hw" / "nvdla" / "mapspace" / "mapspace.yaml",
                        prob.path
                    ]
            elif arch_name == "gemmini":
                    paths = [ # Gemmini
                        DATASET_ROOT_PATH / "hw" / "gemmini" / "arch" / "arch.yaml",
                        DATASET_ROOT_PATH / "hw" / "gemmini" / "mapspace" / "mapspace_arch_baseline.yaml",
                        prob.path
                    ]
            output_path = output_dir / arch_name / workload_name / prob.config_str()
            output_path.mkdir(parents=True, exist_ok=True)
            output_paths.append(output_path)
            with open(output_path / "random.txt", "w") as f:
                p = run_timeloop(paths, stdout=f, cwd=output_path, run_async=True)
                procs.append(p)
        while any([p.poll() is None for p in procs]):
            time.sleep(1)
        for layer_idx, prob in enumerate(layers):
            rows = parse_timeloop_output(output_paths[layer_idx], prob, return_min_fn=lambda row: row["target.edp"])
            cycle_total += rows[0]["target.cycle"] * layer_count[layer_idx]
            energy_total += rows[0]["target.energy"] * layer_count[layer_idx]
        logger.info(arch_name)
        logger.info(workload)
        logger.info("Total cycle: %s", cycle_total)
        logger.info("Total energy: %s", energy_total)
        logger.info("Total EDP: %s", cycle_total * energy_total)
        results[arch_name] = cycle_total * energy_total

    plt.rcParams['figure.figsize'] = 5.3, 5.3
    # plt.rc('xtick', labelsize=12)
    plt.figure()
    fig, ax = plt.subplots()
    plt.title(workload)
    plt.ylabel("Energy-delay product (uJ x cycles)")
    arch_names = ["Eyeriss", "NVDLA\nSmall", "NVDLA\nLarge", "Gemmini\nDefault", "Gemmini\nDOSA"]
    arch_perfs = [results[arch] for arch in results.keys()] + [gd_result]
    bar_colors = ["indianred", "lightgreen", "mediumseagreen", "lightblue", "tab:blue"]
    # bar_colors = ["gray", "gray", "gray", "gray", "tab:blue"]
    bars = ax.bar(arch_names, arch_perfs, color = bar_colors, edgecolor="black")
    relative_arch_perf = [str(round(arch_perfs[i]/gd_result, 1)) + "x" for i in range(len(arch_perfs))]
    ax.bar_label(bars, labels=relative_arch_perf)
    ax.set_ylim(0, 1.05*ax.get_ylim()[1])
    # plt.xticks([])
    # plt.tight_layout()
    plt.savefig(output_dir / f"arch_compare_{workload}.png", bbox_inches="tight")

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
    return parser

if __name__ == "__main__":
    args = construct_argparser().parse_args()
    main(args.output_dir, args.workload)

# arch_baseline_results = {
#     "Eyeriss": {
#         "ResNet-50":  780976808884.6254,
#         "BERT": 19719346875.711487,
#         "RetinaNet":  488501424178.48114,
#         "U-Net": 1108363658717171.8,
#     },
#     "NVDLA\nSmall": {
#         "ResNet-50": 1791713501954.344,
#         "BERT": 73535003875.49184,
#         "RetinaNet": 915733179419.7189,
#         "U-Net": 2239039362066733.8,
#     },
#     "NVDLA\nLarge": {
#         "ResNet-50": 213346427199.6543,
#         "BERT": 6845260581.061227,
#         "RetinaNet": 107743771785.74187,
#         "U-Net": 141941040848882.53,
#     },
#     "Gemmini\nDefault": {
#         "ResNet-50": 250199353098.03815,
#         "BERT": 9085864499.35872,
#         "RetinaNet": 146460188214.4036,
#         "U-Net": 252173694896890.3,
#     }
# }
