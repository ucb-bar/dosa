import collections
import errno
import json
import logging
import math
import os
import pathlib
import pickle
import random
import re
import string
import subprocess
import sys
import tarfile
from time import strftime, gmtime
from typing import List

import numpy as np
import yaml
import torch

from dataset.common import logger

_TIMELOOP_MODEL_PATH = "timeloop-model"
_TIMELOOP_MAPPER_PATH = "timeloop-mapper"


class DatasetSeed():
    def __init__(self, seed):
        self.seed = seed
dataset_seed = DatasetSeed(0)
def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    dataset_seed.seed = random_seed
def get_random_seed():
    return dataset_seed.seed


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class OrderedDefaultdict(collections.OrderedDict):
    """ A defaultdict with OrderedDict as its base class. """

    def __init__(self, default_factory=None, *args, **kwargs):
        if not (default_factory is None
                or isinstance(default_factory, collections.Callable)):
            raise TypeError('first argument must be callable or None')
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory  # called by __missing__()

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key, )
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):  # optional, for pickle support
        args = (self.default_factory,) if self.default_factory else tuple()
        return self.__class__, args, None, None, self.iteritems()

    def __repr__(self):  # optional
        return '%s(%r, %r)' % (self.__class__.__name__, self.default_factory,
                               list(self.iteritems()))


def delete_dramsim_log(path=pathlib.Path('.')):
    files = path.glob('dramsim.*.log')
    for f in files:
        f.unlink()


def setup_logging(module_name, logger):
    # logging setup
    def logfilename():
        """ Construct a unique log file name from: date + 16 char random. """
        timeline = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
        randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        return module_name + "-" + timeline + "-" + randname + ".log"

    # log to file
    full_log_filename = logfilename()
    fileHandler = logging.FileHandler(full_log_filename)
    # formatting for log to file
    # TODO: filehandler should be handler 0 (firesim_topology_with_passes expects this
    # to get the filename) - handle this more appropriately later
    logFormatter = logging.Formatter("%(asctime)s [%(funcName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.NOTSET)  # log everything to file
    logger.addHandler(fileHandler)

    # log to stdout, without special formatting
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)  # show only INFO and greater in console
    logger.addHandler(consoleHandler)


def dict_append_val(d, k, v):
    if k in d.keys():
        d[k].append(v)
    else:
        d[k] = [v]


def gen_makefile(x_size, y_size):
    makefile_str = """EXT_DRAMSIM2_PATH = ../DRAMSim2
LIBS = -ldramsim

DEBUG_LEVEL ?= 1

USER_FLAGS += -DDISABLE_PACER -DDEBUG_LEVEL=$(DEBUG_LEVEL) -std=c++11 \
-I$(EXT_DRAMSIM2_PATH) -L$(EXT_DRAMSIM2_PATH) $(LIBS) -DNOC_X={} -DNOC_Y={}

include ../unittests_Makefile
""".format(x_size, y_size)

    with open("Makefile", "w") as f:
        f.write(makefile_str)


def compile_sim():
    # subprocess.check_call(['make', 'clean'],\
    #        cwd=os.getcwd())
    orig_sim_test = pathlib.Path('sim_test')
    if orig_sim_test.exists():
        orig_sim_test.unlink()
    try:
        out = subprocess.check_output(['make'],
                                      stderr=subprocess.PIPE, cwd=os.getcwd())

        logger.info('compile_sim> make')
    except subprocess.CalledProcessError as grepexc:
        # print "error code", grepexc.returncode, grepexc.output
        if grepexc.returncode == 2:
            pass
            # logger.error(err)


def run_sim(prefix, stdout=None, stderr=None, timeout=None, exe='sim_test', nb=False):
    print("./{} {}".format(exe, prefix))
    try:
        exe = str(exe)
        if nb:
            p = subprocess.check_call(['./' + exe, str(prefix)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, cwd=os.getcwd(),
                                      timeout=timeout)
        else:
            p = subprocess.check_call(['./' + exe, str(prefix)],
                                      stdout=stdout, stderr=stderr, cwd=os.getcwd(), timeout=timeout)
        logger.info('run_sim> ./' + exe + ' ' + str(prefix))
        return True
    except Exception:
        return False


def run_cosa(arch_path, prob_path, mapping_path, output_dir, output_mapper_yaml_path, cwd=os.getcwd(), stdout=None, stderr=None):
    cmd = f'python src/cosa.py -ap {arch_path} -pp {prob_path} -mp {mapping_path} -o {output_dir} -omap {output_mapper_yaml_path}'
    args = cmd.split(' ')
    print(cmd)
    logger.info('run_cosa> {}'.format(cmd))
    try:
        p = subprocess.check_call(args, cwd=cwd, stdout=None, stderr=stderr)
        return True
    except Exception as e:
        logger.error("Failed to run python cosa.py with exception %s", e)
        return False

def run_cosa_async(arch_path, prob_path, mapping_path, output_dir, output_mapper_yaml_path, cwd=os.getcwd(), stdout=None, stderr=None):
    cmd = f'python src/cosa.py -ap {arch_path} -pp {prob_path} -mp {mapping_path} -o {output_dir} -omap {output_mapper_yaml_path}'
    args = cmd.split(' ')
    print(cmd)
    logger.info('run_cosa_async> {}'.format(cmd))
    try:
        p = subprocess.Popen(args, cwd=cwd, stdout=None, stderr=stderr)
        return p
    except Exception as e:
        logger.error("Failed to run python cosa.py with exception %s", e)
        return None

def run_timeloop(arch, prob, mapping, cwd=os.getcwd(), stdout=None, stderr=None):
    arch_path = pathlib.Path(arch).resolve()
    prob_path = pathlib.Path(prob).resolve()
    mapping_path = pathlib.Path(mapping).resolve()
    args = [_TIMELOOP_MODEL_PATH, arch_path, mapping_path, prob_path, ]
    logger.info('run_timeloop> timeloop-model {} {} {}'.format(*args))
    try:
        p = subprocess.check_call(args, cwd=cwd, stdout=stdout, stderr=stderr)
        return True
    except Exception as e:
        logger.error("Failed to run timeloop-model with exception %s", e)
        return False


def run_timeloop_mapper(arch, prob, mapspace, cwd=os.getcwd(), stdout=None, stderr=None, run_async=False):
    arch_path = pathlib.Path(arch).resolve()
    prob_path = pathlib.Path(prob).resolve()
    mapspace_path = pathlib.Path(mapspace).resolve()
    args = [_TIMELOOP_MAPPER_PATH, arch_path, mapspace_path, prob_path, ]
    logger.info('run_timeloop> {} {} {} {}'.format(*args))
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


def run_timeloop_mapper_optimal(arch, prob, mapspace, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        arch_path = pathlib.Path('timeloop_configs/arch/simba_large.yaml').resolve()
        mapspace_path = pathlib.Path('timeloop_configs/mapspace/mapspace_io_optimal.yaml').resolve()
        p = subprocess.check_call(['timeloop-mapper', arch_path, mapspace_path, str(prob), ],
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_timeloop> timeloop-mapper {} {} {}'.format(arch, mapspace, prob))
        return True
    except Exception as e:
        logger.error("Failed to run timeloop-mapper with exception %s", e)
        return False


def run_timeloop_mapper_linear(arch, prob, mapspace, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        print("Running LINEAR PRUNING")
        arch_path = pathlib.Path('timeloop_configs/arch/simba_large.yaml').resolve()
        mapspace_path = pathlib.Path('timeloop_configs/mapspace/mapspace_io_linear.yaml').resolve()
        p = subprocess.check_call(['timeloop-mapper', arch_path, mapspace_path, str(prob), ],
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_timeloop> timeloop-mapper {} {} {}'.format(arch, mapspace, prob))
        return True
    except Exception as e:
        logger.error("Failed to run timeloop-mapper with exception %s", e)
        return False


def run_timeloop_mapper_hybrid(arch, prob, mapspace, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        print("Running HYBRID Random Index + Linear Pruning")
        arch_path = pathlib.Path('timeloop_configs/arch/simba_large.yaml').resolve()
        mapspace_path = pathlib.Path('timeloop_configs/mapspace/mapspace_io_hybrid.yaml').resolve()
        p = subprocess.check_call(['timeloop-mapper', arch_path, mapspace_path, str(prob), ],
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_timeloop> timeloop-mapper {} {} {}'.format(arch, mapspace, prob))
        return True
    except Exception as e:
        logger.error("Failed to run timeloop-mapper with exception %s", e)
        return False


def read_file(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    return data


def store_json(json_path, data, indent=None):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)


def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data


def parse_csv(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines: 
            line = line.strip()
            line_arr = line.split(',') 
            data.append(line_arr)
    return data 


def parse_csv_line(csv_path, line_idx):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        line = lines[line_idx]
        line = line.strip()
        line_arr = line.split(',') 
    return line_arr 


def store_yaml(yaml_path, data):
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)


def parse_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def store_pickle(pickle_path, data):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


# Return prime values from small to large
def get_prime_factors(value):
    factors = []

    # Print the number of two's that divide n 
    while (value % 2 == 0):
        factors.append(2)
        value = value // 2

    # value must be odd at this point 
    # so a skip of 2 ( i = i + 2) cavalue be used 
    for i in range(3, int(math.sqrt(value)) + 1, 2):

        # while i divides value , print i ad divide value 
        while (value % i == 0):
            factors.append(i)
            value = value // i

            # Condition if value is a prime
    # number greater than 2 
    if value > 2:
        factors.append(value)
    if len(factors) == 0:
        factors.append(1)

    return factors


def get_divisors(value: int) -> list[int]:
    res = list()
    i = 1
    while i <= value : 
        if (value % i==0) : 
            res.append(i), 
        i = i + 1
    return res

def get_nearest_choice(value: float, sorted_choices: list[int]) -> int:
    for idx, choice in enumerate(sorted_choices):
        if value < choice:
            if idx == 0: return choice
            if (choice-value)>(value - sorted_choices[idx-1]):
                return sorted_choices[idx-1]
            return choice
    return sorted_choices[-1]

def round_down_choices(value: float, sorted_choices: list[int]) -> int:
    for idx, choice in enumerate(sorted_choices):
        if value < choice:
            if idx == 0: return choice
            if value >= sorted_choices[idx-1]:
                return sorted_choices[idx-1]
    return sorted_choices[-1]

def get_whole_ceil(n,near) -> int:
    nn = np.divide(n,np.linspace(1,np.ceil(n/near),int(np.ceil(n/near))))
    return int(nn[nn%1==0][-1])

def get_whole_floor(n,near):
    nn = np.divide(n,np.linspace(np.floor(n/near),n,int(n-np.floor(n/near)+1)))
    return int(nn[nn%1==0][0])


# remove factor from prime factors
# prob_factors is passed by ref
def update_prime_factors(prob_factors, prob_idx, factor):
    logger.debug("prob_factors {}".format(prob_factors))
    rm_idx = []
    orig_factor = factor
    # print(prob_factors)
    for i, val in enumerate(prob_factors[prob_idx]):
        if factor % val == 0:
            rm_idx.append(i)
            factor = factor // val
        if factor == 1:
            break

    vals = []
    rm_idx.reverse()
    for i in rm_idx:
        vals.append(prob_factors[prob_idx].pop(i))

    if len(prob_factors[prob_idx]) == 0:
        prob_factors[prob_idx] = [1]
    pop_val = 1
    # print(prob_factors)
    logger.debug("updated_prob_factors {}".format(prob_factors))
    for i in vals:
        pop_val = pop_val * i
    assert (pop_val == orig_factor)


# shrink the space
def shrink_factor_space(prob_factors):
    # val get rid of factor of 1
    for prob_idx, _ in enumerate(prob_factors):
        if len(prob_factors[prob_idx]) == 1:
            if prob_factors[prob_idx][0] == 1:
                prob_factors[prob_idx] = []


def compose_prob(file_path, prob_dict):
    input_dict = {}
    input_dict["problem"] = prob_dict
    input_dict['problem']['shape'] = 'cnn-layer'
    input_dict['problem']['Wstride'] = 1
    input_dict['problem']['Hstride'] = 1
    input_dict['problem']['Wdilation'] = 1
    input_dict['problem']['Hdilation'] = 1
    store_yaml(file_path, input_dict)


def get_perm_arr_from_val(val, perm_arr_tup, prob_levels):
    order_idx = np.unravel_index(val, perm_arr_tup)
    perm_idx = list(range(prob_levels))
    perm_arr = [-1] * len(perm_arr_tup)
    # idx is the idx into [0,1,2,3..,6]
    for i, idx in enumerate(order_idx):
        perm_arr[i] = perm_idx[idx]
        del perm_idx[idx]
    return perm_arr


# input perm_arr [0, 3, 5, 6, 2, 4, 1] -> order_idx 
# order_idx -> val 333 
def get_val_from_perm_arr(self, perm_arr, perm_arr_tup, prob_levels):
    assert (len(perm_arr) == len(perm_arr_tup))
    order_idx = [0] * len(perm_arr)
    perm_idx = list(range(prob_levels))
    for i, idx in enumerate(perm_arr):
        for j, val in enumerate(perm_idx):
            if val == idx:
                order_idx[i] = j
                del perm_idx[j]
                break
    val = np.ravel_multi_index(tuple(order_idx), perm_arr_tup)
    return val


def to_config_str_key(configs):
    configs_arr = [str(v) for v in configs]
    configs_str = "_".join(configs_arr)
    return configs_str


def get_correlation(a, b):
    a = np.array(a)
    b = np.array(b)
    cor = np.corrcoef(a, b)[0, 1]
    return cor


def get_area_stats(stats_txt_file):
    with open(stats_txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        m = re.match(r"Area: (.*) mm\^2", line)
        if m:
            area = float(m.group(1))
    return area


def get_cor_stats(status_dicts):
    arr = []
    for d in status_dicts:
        # baseline
        l_arr = [d['cycle_results'][0]]
        l_arr.append(d['cycle_results'][1])
        l_arr.append(d['cycle_results'][2])
        l_arr.append(d['cycle_results'][4])
        l_arr.append(d['cycle_results'][5])
        l_arr.append(d['cost']['Total'])
        l_arr.append(d['hop_cost'][0] + d['hop_cost'][1])
        l_arr.append(d['hop_cost'][0])
        l_arr.append(d['hop_cost'][1])
        l_arr.append((d['utilized_capacity'][1] + d['utilized_capacity'][2] + d['utilized_capacity'][3]))
        l_arr.append((d['utilized_capacity'][1] + d['utilized_capacity'][2] + d['utilized_capacity'][3] +
                      d['utilized_capacity'][4]))

        l_arr.append((d['utilized_capacity'][1] * d['utilized_capacity'][2] * d['utilized_capacity'][3]) ** (1 / 128))
        l_arr.append((d['utilized_capacity'][1] * d['utilized_capacity'][2] * d['utilized_capacity'][3] *
                      d['utilized_capacity'][4]) ** (1 / 128))
        l_arr.append((np.log2(1 + d['utilized_capacity'][1]) + np.log2(1 + d['utilized_capacity'][2]) + np.log2(
            1 + d['utilized_capacity'][3])))
        l_arr.append((np.log2(1 + d['utilized_capacity'][1]) + np.log2(1 + d['utilized_capacity'][2]) + np.log2(
            1 + d['utilized_capacity'][3]) + np.log2(1 + d['utilized_capacity'][4])))

        arr.append(l_arr)
    arr_np = np.array(arr).T

    cors = []
    for i in range(1, len(l_arr)):
        cor = get_correlation(arr_np[0, :], arr_np[i, :])
        cors.append(cor)

    print(arr_np[-3, :])
    x_label = "Total_Cycles"
    print(f"{x_label}-Unicast_Cycles: {cors[0]}")
    print(f"{x_label}-Multicast_Cycles: {cors[1]}")
    print(f"{x_label}-DRAM_READ_Cycles: {cors[2]}")
    print(f"{x_label}-DRAM_WRITE_Cycles: {cors[3]}")
    print(f"{x_label}-Cost: {cors[4]}")
    print(f"{x_label}-Total_Hops: {cors[5]}")
    print(f"{x_label}-Unicast_Hops: {cors[6]}")
    print(f"{x_label}-Multicast_Hops: {cors[7]}")
    print(f"{x_label}-BufSumGBEx: {cors[8]}")
    print(f"{x_label}-BufSumGBIn: {cors[9]}")
    print(f"{x_label}-BufProdGBEx ** (1/128): {cors[10]}")
    print(f"{x_label}-BufProdGBIn ** (1/128): {cors[11]}")
    print(f"{x_label}-BufProdGBEx log): {cors[12]}")
    print(f"{x_label}-BufProdGBIn log: {cors[13]}")

    return cors


def get_invalid_samples(temp_file):
    total_invalid = 0
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        m = re.match(r".*MATCHLIB_TOTAL: (.*), MATCHLIB_VALID.*", line)
        if m:
            invalid = m.group(1)
            total_invalid += int(invalid)
    return total_invalid


def get_runtime(summary_out):
    runtime_dict = {}

    with open(summary_out, 'r') as f:
        lines = f.readlines()
    key = None
    for line in lines:
        m = re.match(r".*simba_(.*)\/(.*)$", line)
        if m:
            key = m.group(2)
        else:
            m = re.match(r"Elasped time for .*find solution with .* is: (.*)$", line)
            if m:
                runtime_dict[key] = float(m.group(1))
    return runtime_dict


def update_arch(arch_dict, mem_entries, out_path):
    for i, mem in enumerate(arch_dict['arch']['storage']):
        # DRAM has no entries
        if i < len(arch_dict['arch']['storage']) - 1:
            arch_dict['arch']['storage'][i]['entries'] = mem_entries[i]
    store_yaml(out_path, arch_dict)


def unique_filename(extension: str, prefix: str = ""):
    """
    Construct a unique file name from: extension (without dot), date + 16 char random, and optional prefix
    """
    timeline = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    filename = timeline + "-" + randname
    if extension:
        filename = filename + "." + extension
    if prefix:
        filename = prefix + "-" + filename
    return filename


def num_mem_lvls(arch_name: str):
    if arch_name == "gemmini":
        return 4
    if arch_name == "simba":
        return 6


"""DSE util functions"""


def keys_by_type(df, key_type: str, scalar_only: bool = True) -> List[str]:
    """Get DataFrame column headers or dict keys of a certain type.

    key_types = ['target', 'arch', 'mapping', 'prob']

    scalar_only: keys with scalar values only (exclude mapping.mapping)
        True by default.
    """
    scalar_exclude = {"mapping.mapping", "mapping.flat_mapping", "arch.name", "prob.shape", "prob.shape_cls"} # set type
    keys = []
    for key in df.keys():
        if str(key).startswith(key_type):
            if scalar_only:
                if key in scalar_exclude:
                    continue
                if not np.isscalar(df.loc[0, key]):
                    continue
            keys.append(key)
    return keys

def get_prob_dims(prob_shape: str) -> List[str]:
    """Get names of problem dimensions for a given problem shape.
    """
    dims = []
    if prob_shape == "cnn-layer":
        dims = ["R", "S", "P", "Q", "C", "K", "N"]
    else:
        logger.error("Unsupported problem shape %s", prob_shape)
    return dims

prob_shape_cls_map = {
    "cnn-layer": 1
}
prob_cls_shape_map = {v: k for k, v in prob_shape_cls_map.items()}

def prob_shape_to_class(prob_shape: str) -> int:
    """Converts problem type to integer representation

    Args:
        prob_shape (str): [cnn-layer]
    
    Returns:
        Integer representing mapped problem type
    """
    return prob_shape_cls_map[prob_shape]

def prob_class_to_shape(prob_cls: int) -> str:
    """Converts problem type to integer representation

    Args:
        prob_shape (str): [cnn-layer]
    
    Returns:
        Integer representing mapped problem type
    """
    return prob_cls_shape_map[prob_cls]

def search_curve(vals):
    best_per_point = []
    best_so_far = float("inf")
    for val in vals:
        if val < best_so_far:
            best_so_far = val
        best_per_point.append(best_so_far)
    return best_per_point
