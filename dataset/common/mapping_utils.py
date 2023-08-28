import re
import copy
import math
import functools

import numpy as np
import torch

from dataset.common import logger, utils
from dataset.workloads import Prob
from dataset.dse import pytorch_util

tensor_to_dims = {
    0: {"R", "S", "C", "K"}, # W
    1: {"P", "Q", "C", "N"}, # I
    2: {"P", "Q", "K", "N"}, # O
}

def process_mapping(mapping_str: str, prob_shape: str) -> list[np.float64]:# -> Tuple[List[int], List[str]]:
    """Converts shorthand mapping to matrix representation

    shape,C,Hdilation,Hstride,K,N,P,Q,R,S,Wdilation,Wstride
    cnn-layer,64,1,1,64,1,56,56,1,1,1,1

    {R: 1, S: 1, P: 56, Q: 56, C: 64, K: 64, N: 1}

    Args:
        mapping_str (str): Compressed string representation of mapping, e.g.
            L3[WIO] C8 Q7 K16 - L2[WI] Q2 C2 - L1[O] P7 K2 C4 K2X - L0[W] Q4 P8

    Returns:
        Flattened array representation of tiling factors. Like the below
        example for CNN layers, except flattened rather than nested.

            [memlvl3, memlvl2, ..., memlvl0]
              \\  /
               \\/
                [spatial tile, temporal tile, permutation]
                    \\  /
                     \\/
                     [R, S, P, Q, C, K, N]

        Permutation is only for temporal tiling factors. Permutation 
        example for L3[WIO] C8 Q7 K16, given same ordering as above:

            [memlvl3, memlvl2, memlvl1, memlvl0]
              \\  /
               \\/
                [spatial tile, temporal tile, permutation]
                                                \\  /
                                                 \\/
                                                 [7, 7, 7, 5, 6, 4, 7]
    """
    # Get string for each mem lvl
    mapping_by_lvl = mapping_str.split(" - ")
    num_mem_lvls = len(mapping_by_lvl)

    # Get dim names and map to index, e.g. {R:0, S:1, P:2, Q:3, C:4, K:5, N:6}
    dims = utils.get_prob_dims(prob_shape)
    prob_dim_mapping = {dim: i for i, dim in enumerate(dims)}
    num_dims = len(dims)

    # Create flattened array, according to format in docstring
    flat_mapping = []
    for _ in range(num_mem_lvls):
        flat_mapping += [1] * num_dims * 2 # spatial and temporal tiling factors start as 1
        flat_mapping += [num_dims] * num_dims # permutations start as num_dims = don't care

    # parse one lvl at a time
    # perm_step = 0.01
    perm_step = 1
    for inverse_mem_lvl, lvl_str in enumerate(mapping_by_lvl):
        perm_counter = 7 - perm_step # counts down
        # e.g. inverse_mem_lvl = 2, lvl_str = "L1[O] P7 K2 C4 K2X"
        words = lvl_str.split()
        mem_info = words[0] # e.g. L1[O] TODO: use to encode dataflow info
        # print(mem_info)
        # tensors = mem_info.split("[")[1].split("]")[0] # e.g. O
        words = words[1:]
        for word in words: # reverse to get correct permutation - not doing this
            # (dim)(factor)(spatial dim?)
            m = re.match(r"([a-zA-Z]*)([0-9]+)([XY]?)", word)
            if not m:
                logger.error("Failed to match a word in mapping: %s in %s",
                    word, mapping_str)
            dim = m.group(1)
            factor = int(m.group(2))
            spatial = m.group(3)

            # skip any already processed mem lvls, e.g. for lvl 2 skip lvl 3
            offset = num_dims * inverse_mem_lvl * 3
            # set spatial or temporal factor
            if spatial == "X" or spatial == "Y":
                # e.g. for K, skip up to RSPQC
                factor_index = offset + prob_dim_mapping[dim]
            else:
                offset += num_dims # go past spatial factors
                # rest is same logic as spatial
                factor_index = offset + prob_dim_mapping[dim]

                # set perm if temporal (ignore spatial perm)
                # counts up from 1. those with factor 1 remain as num_dims
                perm_index = factor_index + num_dims
                flat_mapping[perm_index] = perm_counter
                perm_counter -= perm_step

            flat_mapping[factor_index] = factor

    return np.array(flat_mapping, dtype=np.float64)#, self.get_mapping_columns(prob_shape, num_mem_lvls)

def get_num_mem_lvls(mapping_str: str) -> int:
    mapping_by_lvl = mapping_str.split(" - ")
    num_mem_lvls: int = len(mapping_by_lvl)
    return num_mem_lvls

def capacity_from_mapping(mapping: list[int] | torch.Tensor, prob: Prob) -> tuple[int, int, list[int]]:
    """
    Return capacity needed per tensor, per level to store a mapping.
    
    Args:
        mapping: flattened mapping, see process_mapping() for format.
        prob: Prob object describing layer

    Returns:
        tuple (mac_needed, max_spatial_factor, buf_needed)
        mac_needed is number of MAC units needed to satisfy spatial tiling factors
        max_spatial_factor is equal to greatest individual spatial factor in mapping
        buf_needed stores needed capacity per buffer, in words
            [[W, I, O], [W, I, O], ...]
    """
    # Get dim names and map to index, e.g. {R:0, S:1, P:2, Q:3, C:4, K:5, N:6}
    all_dims = utils.get_prob_dims(prob.shape)
    dim_idx_dict = prob.prob_name_idx_dict
    num_dims = len(all_dims)
    num_mem_lvls = len(mapping) // num_dims // 3
    mac_needed = 1
    max_spatial_factor = torch.tensor(1.).to(pytorch_util.device)
    buf_needed = [[torch.tensor(1.).to(pytorch_util.device), torch.tensor(1.).to(pytorch_util.device), torch.tensor(1.).to(pytorch_util.device)] for _ in range(num_mem_lvls)] # copy by value
    p_edge_mem_lvl = -1
    q_edge_mem_lvl = -1
    hstride_mem_lvl = -1
    wstride_mem_lvl = -1

    # copy registers factors to acc level GEMMINI SPECIFIC
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.tensor(mapping)
    reg_start_idx = (num_mem_lvls - 1) * num_dims * 3 + num_dims + 2 # only P and Q temporal factors
    reg_end_idx = reg_start_idx + 2
    reg_factors = mapping[reg_start_idx:reg_end_idx]
    acc_start_idx = (num_mem_lvls - 2) * num_dims * 3 + num_dims + 2
    acc_end_idx = acc_start_idx + 2
    acc_factors = mapping[acc_start_idx:acc_end_idx]
    # for fac_idx, map_idx in enumerate(range(acc_start_idx, acc_end_idx)):
    #     mapping[map_idx] = reg_factors[fac_idx] * acc_factors[fac_idx]
    new_acc_factors = acc_factors*reg_factors
    mapping = torch.cat((mapping[:acc_start_idx], new_acc_factors, mapping[acc_end_idx:]))

    for mem_lvl in range(1, num_mem_lvls):
        # # start by initializing space needed to that needed by previous level
        # if mem_lvl > 0:
        #     buf_needed[mem_lvl] = buf_needed[mem_lvl - 1][:]
        
        total_p = 1
        total_q = 1
        total_r = 1
        total_s = 1

        for inner_mem_lvl in range(1, mem_lvl+1):
            # get mapping for that level
            start_idx = (num_mem_lvls - 1 - inner_mem_lvl) * num_dims * 3
            end_idx = start_idx + num_dims * 3
            mem_lvl_mapping = mapping[start_idx:end_idx]
            spatial_factors = mem_lvl_mapping[:num_dims]
            temporal_factors = mem_lvl_mapping[num_dims:num_dims*2]

            # spatial factors
            for dim in dim_idx_dict: # check all dims
                sf = spatial_factors[dim_idx_dict[dim]]
                if sf > 1:
                    mac_needed = mac_needed * sf
                    max_spatial_factor = torch.maximum(max_spatial_factor, sf)
                    for tensor, dims in tensor_to_dims.items():
                        if dim in dims:
                            real_sf = sf
                            # if tensor == 1 and dim == "P" and p_edge_mem_lvl==-1: # handle input edges
                            #     real_sf += prob.prob["R"]//2*2
                            #     p_edge_mem_lvl = mem_lvl
                            # elif tensor == 1 and dim == "Q" and q_edge_mem_lvl==-1: # handle input edges
                            #     real_sf += prob.prob["S"]//2*2
                            #     q_edge_mem_lvl = mem_lvl
                            buf_needed[mem_lvl][tensor] = buf_needed[mem_lvl][tensor] * real_sf
                            # if mem_lvl == num_mem_lvls - 1: # only multiply spatial factors once to get registers capacity
                            #     buf_needed[0][tensor] = buf_needed[0][tensor] * real_sf

            # temporal factors
            for tensor, relevant_dims in tensor_to_dims.items():
                for dim in dim_idx_dict: # only check the relevant dims
                    tf = temporal_factors[dim_idx_dict[dim]]
                    if tf > 1:
                        if tensor == 1:
                            # multiply by stride for inputs
                            # NOTE: may not be accurate for all mem lvls, in particular lvl 0
                            # if dim == "P" and wstride_mem_lvl == -1:
                            #     tf = tf * prob.prob.get("Wstride", 1)
                            #     wstride_mem_lvl = mem_lvl
                            # elif dim == "Q" and hstride_mem_lvl == -1:
                            #     tf = tf * prob.prob.get("Hstride", 1)
                            #     hstride_mem_lvl = mem_lvl
                            # if dim == "P" and p_edge_mem_lvl==-1: # handle input edges
                            #     tf = tf + prob.prob["R"]//2*2
                            #     p_edge_mem_lvl = mem_lvl
                            # elif dim == "Q" and q_edge_mem_lvl==-1: # handle input edges
                            #     tf = tf + prob.prob["S"]//2*2
                            #     q_edge_mem_lvl = mem_lvl
                            if dim == "P":
                                total_p = total_p * tf
                                continue # don't multiply this factor - handled later
                            elif dim == "Q":
                                total_q = total_q * tf
                                continue
                            elif dim == "R":
                                total_r = total_r * tf
                            elif dim == "S":
                                total_s = total_s * tf
                        if dim in relevant_dims:
                            buf_needed[mem_lvl][tensor] = buf_needed[mem_lvl][tensor] * tf

        buf_needed[mem_lvl][1] = buf_needed[mem_lvl][1] * ((total_p-1) * prob.prob.get("Wstride", 1) + total_r)
        buf_needed[mem_lvl][1] = buf_needed[mem_lvl][1] * ((total_q-1) * prob.prob.get("Hstride", 1) + total_s)
    return mac_needed, max_spatial_factor, buf_needed

def mapping_index(num_mem_lvls: int, num_dims: int, mem_lvl: int, type: str, offset: int):
    type_order = {
        "spatial": 0,
        "temporal": 1,
        "perm": 2,
    }
    idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3 + type_order[type] * num_dims + offset
    return idx

def accesses_from_mapping(mapping: list[int] | torch.Tensor, prob: Prob, with_cache: bool = False) -> list[list[int]]:
    """Calculates number of reads, updates, writes to each memory level

    Not accurate for bypassed tensors at each memory level
    """
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.tensor(mapping)
    # Get dim names and map to index, e.g. {R:0, S:1, P:2, Q:3, C:4, K:5, N:6}
    dim_idx_dict = prob.prob_name_idx_dict
    idx_dim_dict = prob.prob_idx_name_dict
    num_dims = len(dim_idx_dict)
    num_mem_lvls = len(mapping) // num_dims // 3

    # # copy registers factors to acc level GEMMINI SPECIFIC
    # reg_start_idx = (num_mem_lvls - 1) * num_dims * 3
    # reg_end_idx = reg_start_idx + num_dims * 2
    # reg_factors = mapping[reg_start_idx:reg_end_idx]
    # acc_start_idx = (num_mem_lvls - 2) * num_dims * 3
    # acc_end_idx = acc_start_idx + num_dims * 2
    # acc_factors = mapping[acc_start_idx:acc_end_idx]
    # new_acc_factors = acc_factors*reg_factors
    # new_reg_factors = reg_factors[:]
    # for idx in range(len(new_reg_factors)):
    #     new_reg_factors[idx] = 1.
    # mapping = torch.cat((mapping[:acc_start_idx], 
    #                      new_acc_factors,
    #                      mapping[acc_end_idx:reg_start_idx],
    #                      new_reg_factors,
    #                      mapping[reg_end_idx:]))

    _, _, capacities = capacity_from_mapping(mapping, prob)
    writes = [[0, 0, 0] for _ in range(num_mem_lvls)]
    for mem_lvl in range(num_mem_lvls-1):
        for tensor in [0, 2]: # inputs handled differently
            writes[mem_lvl][tensor] = capacities[mem_lvl][tensor]
        writes[mem_lvl][1] = 1

    local_tensor_to_dims = copy.deepcopy(tensor_to_dims)
    local_tensor_to_dims[1].update({"R", "S"})

    Wstride = prob.prob.get("Wstride", 1)
    Hstride = prob.prob.get("Hstride", 1)
    for mem_lvl in range(num_mem_lvls-1):
        # Handle inputs
        inner = {"P": 1, "Q": 1, "R": 1, "S": 1,}
        for inner_mem_lvl in range(mem_lvl+1):
            for dim in ["P", "Q", "R", "S"]:
                dim_idx = dim_idx_dict[dim]
                idx = mapping_index(num_mem_lvls, num_dims, inner_mem_lvl, "temporal", dim_idx)
                if mapping[idx] > 1:
                    inner[dim] = inner[dim] * mapping[idx]
            for dim in ["C", "N"]:
                if inner_mem_lvl == 0: # Gemmini specific - ignore dims other than P and Q in reg lvl
                    continue
                for factor_type in ["temporal", "spatial"]:
                    dim_idx = dim_idx_dict[dim]
                    idx = mapping_index(num_mem_lvls, num_dims, inner_mem_lvl, factor_type, dim_idx)
                    if mapping[idx] > 1:
                        # print(mem_lvl, inner_mem_lvl, dim, mapping[idx])
                        writes[mem_lvl][1] = writes[mem_lvl][1] * mapping[idx]

        # Handle weights and outputs
        first_outer_input_dim = 1
        first_outer_input_perm_order = 0
        for tensor, relevant_dims in local_tensor_to_dims.items():
            mul_current = False
            for outer_mem_lvl in range(mem_lvl+1, num_mem_lvls): # multiply tiling factors from outer levels
                perm_start_idx = mapping_index(num_mem_lvls, num_dims, outer_mem_lvl, "perm", 0)
                perm_end_idx = perm_start_idx+7
                perm = mapping[perm_start_idx:perm_end_idx]
                sorted_idxs = torch.argsort(perm)

                # multiply by any factors that are outer to a relevant factor
                for perm_order, sorted_idx in enumerate(sorted_idxs):
                    if perm_order == 0 and outer_mem_lvl == mem_lvl + 1 and tensor == 1:
                        # need to check which input-related dim is in the innermost loop
                        input_dims_test = {
                            "P": perm[dim_idx_dict["P"]],
                            "Q": perm[dim_idx_dict["Q"]],
                            "R": perm[dim_idx_dict["R"]],
                            "S": perm[dim_idx_dict["S"]],
                        }
                        innermost_input_dim = min(input_dims_test, key=input_dims_test.get)

                        if sorted_idxs[0] == dim_idx_dict["R"] or sorted_idxs[0] == dim_idx_dict["S"]:
                            # an R or S in the first outside loop acts like an inner R or S
                            dim = idx_dim_dict[int(sorted_idxs[0])]
                            idx = mapping_index(num_mem_lvls, num_dims, outer_mem_lvl, "temporal", int(sorted_idxs[0]))
                            inner[dim] = inner[dim] * mapping[idx]
                            if (sorted_idxs[1] == dim_idx_dict["P"] and sorted_idxs[0] == dim_idx_dict["R"]) or \
                                    (sorted_idxs[1] == dim_idx_dict["Q"] and sorted_idxs[0] == dim_idx_dict["S"]):
                                first_outer_input_perm_order = 1
                                innermost_input_dim = idx_dim_dict[int(sorted_idxs[1])]
                                continue

                    sorted_dim = idx_dim_dict[int(sorted_idx)]
                    dim_idx = dim_idx_dict[sorted_dim]
                    if mem_lvl == 0 or tensor == 2:
                        factor_types = ["temporal", "spatial"]
                    else:
                        factor_types = ["temporal"]
                    for factor_type in factor_types:
                        idx = mapping_index(num_mem_lvls, num_dims, outer_mem_lvl, factor_type, dim_idx)
                        # print(mem_lvl, tensor, writes[mem_lvl][tensor], mapping[idx])
                        if outer_mem_lvl == mem_lvl+1 and tensor == 1 and first_outer_input_dim <= 1 and \
                                        mapping[idx] > 1 and sorted_dim == innermost_input_dim and (perm_order == first_outer_input_perm_order):
                            first_outer_input_dim = mapping[idx]
                            mul_current = True
                            continue
                        if mapping[idx] > 1:
                            if factor_type == "temporal":
                                if mul_current == False and sorted_dim in relevant_dims:
                                    mul_current = True
                                elif mul_current == False:
                                    continue
                            # if tensor == 1:
                            #     print(mem_lvl, outer_mem_lvl, sorted_dim, mapping[idx])
                            writes[mem_lvl][tensor] = writes[mem_lvl][tensor] * mapping[idx]
        if innermost_input_dim == "P":
            Wend = 1
            if inner["R"] > 1:
                Wend = Wstride
                if first_outer_input_perm_order == 1:
                    if inner["P"] <= 1:
                        Wend = inner["R"]
                    elif inner["P"] <= inner["R"]:
                        Wend = inner["R"] - 1
                    else:
                        Wend = max(Wstride, inner["R"])
            iters = (((inner["P"] - 1) * Wstride) + (Wend)) * (((inner["Q"] - 1) * Hstride) + inner["S"]) * (first_outer_input_dim-1)
        elif innermost_input_dim == "Q":
            Hend = 1
            if inner["S"] > 1:
                Hend = Hstride
                if first_outer_input_perm_order == 1: # if Q is 1, inner["S"]
                    if inner["Q"] <= 1:
                        Hend = inner["S"]
                    elif inner["Q"] <= inner["S"]:
                        Hend = inner["S"] - 1
                    else:
                        Hend = max(Hstride, inner["S"])
            iters = (((inner["P"] - 1) * Wstride) + inner["R"]) * (((inner["Q"] - 1) * Hstride) + (Hend)) * (first_outer_input_dim-1)
        else:
            iters = 0
        last_iter = (((inner["P"] - 1) * Wstride) + inner["R"]) * (((inner["Q"] - 1) * Hstride) + inner["S"])
        # if mem_lvl == 2:
        #     import pdb
        #     pdb.set_trace()
        writes[mem_lvl][1] = writes[mem_lvl][1] * (iters + last_iter)

    present = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
    reads = [[0, 0, 0] for _ in range(num_mem_lvls)]
    for mem_lvl in range(num_mem_lvls):
        for tensor in range(len(tensor_to_dims)):
            reads[mem_lvl][tensor] = writes[mem_lvl][tensor]

    # register weight reads
    # for dim, dim_idx in dim_idx_dict.items():
    #     idx = mapping_index(num_mem_lvls, num_dims, 0, "temporal", dim_idx)
    #     if mapping[idx] > 1:
    #         reads[0][0] = reads[0][0] * mapping[idx]
    stop_mul = False
    for mem_lvl in range(num_mem_lvls): # multiply tiling factors from outer levels
        perm_start_idx = mapping_index(num_mem_lvls, num_dims, mem_lvl, "perm", 0)
        perm_end_idx = perm_start_idx+7
        perm = mapping[perm_start_idx:perm_end_idx]
        sorted_idxs = torch.argsort(perm)
        for sorted_idx in sorted_idxs:
            sorted_dim = idx_dim_dict[int(sorted_idx)]
            dim_idx = dim_idx_dict[sorted_dim]
            idx = mapping_index(num_mem_lvls, num_dims, mem_lvl, "temporal", dim_idx)
            if mapping[idx] > 1:
                if sorted_dim == "P" or sorted_dim == "Q":
                    reads[0][0] = reads[0][0] * mapping[idx]
                else:
                    stop_mul = True
                    break
        if stop_mul:
            break

    # DRAM weight and input reads, Scratchpad weight reads
    for tensor in [0, 1]:
        for mem_lvl in range(1, num_mem_lvls):
            for inner_mem_lvl in range(mem_lvl-1, -1, -1):
                if present[inner_mem_lvl][tensor]:
                    reads[mem_lvl][tensor] = writes[inner_mem_lvl][tensor]
                    break
    # scratchpad input reads - GEMMINI SPECIFIC
    # equal to register weight reads / K spatial tiling
    idx = mapping_index(num_mem_lvls, num_dims, 2, "spatial", dim_idx_dict["K"])
    reads[num_mem_lvls-2][1] = reads[0][0] / mapping[idx]

    updates = [[0, 0, 0] for _ in range(num_mem_lvls)]
    # acc output updates - GEMMINI SPECIFIC
    # reg weight reads / C spatial tiling
    idx = mapping_index(num_mem_lvls, num_dims, 1, "spatial", dim_idx_dict["C"])
    updates[1][2] = reads[0][0] / mapping[idx]

    # DRAM output updates - GEMMINI SPECIFIC
    # equal to acc writes
    updates[-1][2] = writes[1][2]

    # output reads - equal to updates minus output size (timeloop ENABLE_FIRST_READ_ELISION option)
    output_size = prob.prob["P"] * prob.prob["Q"] * prob.prob["K"] * prob.prob["N"]
    for mem_lvl in range(num_mem_lvls):
        reads[mem_lvl][2] = updates[mem_lvl][2] - output_size

    # print("capacities", capacities)
    # print("writes", writes)
    # print("reads", reads)
    # print("updates", updates)

    # move some reads and updates to cache if we are using cache
    if with_cache:
        cache_size = 512*1024

        tensor_sizes = [functools.reduce(lambda x, y: x * y, [prob.prob[dim] for dim in tensor_to_dims[tensor]]) for tensor in range(3)]
        if sum(tensor_sizes) <= cache_size: # if everything fits in the cache
            cache_reads = [0, 0, 0]
            cache_updates = [0, 0, 0]
            for tensor in range(3):
                if reads[num_mem_lvls-1][tensor] > 0:
                    cache_reads[tensor] = reads[num_mem_lvls-1][tensor] - tensor_sizes[tensor]
                    reads[num_mem_lvls-1][tensor] = tensor_sizes[tensor]
            # output updates
            cache_updates[2] = updates[num_mem_lvls-1][2] - tensor_sizes[2]
            updates[num_mem_lvls-1][2] = tensor_sizes[2]
        else:
            # get dram perm
            perm_start_idx = mapping_index(num_mem_lvls, num_dims, num_mem_lvls-1, "perm", 0)
            perm_end_idx = perm_start_idx+7
            perm = mapping[perm_start_idx:perm_end_idx]
            sorted_idxs = torch.argsort(perm)

            cache_tile_dict = {i:0 for i in range(3)}
            dram_stationary_dict = {i:True for i in range(3)}
            for idx in sorted_idxs:
                new_cache_tile_dict = {i:0 for i in range(3)}
                idx = int(idx)
                dim = idx_dim_dict[idx]
                if dim == "N" or dim == "P" or dim == "Q": # weight stationary
                    new_cache_tile_dict[1] = max(cache_tile_dict[1], tensor_sizes[1])
                    new_cache_tile_dict[2] = max(cache_tile_dict[2], tensor_sizes[2]*4)
                    new_cache_tile_dict[0] = max(cache_tile_dict[0], capacities[2][0])
                    dram_stationary_tensors = {0}
                elif dim == "R" or dim == "S":
                    new_cache_tile_dict[0] = max(cache_tile_dict[0], tensor_sizes[0])
                    new_cache_tile_dict[1] = max(cache_tile_dict[1], capacities[2][1])
                    new_cache_tile_dict[2] = max(cache_tile_dict[2], capacities[1][2]*4)
                    dram_stationary_tensors = {1, 2}
                elif dim == "K": # input stationary
                    new_cache_tile_dict[0] = max(cache_tile_dict[0], tensor_sizes[0])
                    new_cache_tile_dict[2] = max(cache_tile_dict[2], tensor_sizes[2]*4)
                    new_cache_tile_dict[1] = max(cache_tile_dict[1], capacities[2][1])
                    dram_stationary_tensors = {1}
                elif dim == "C": # output stationary
                    new_cache_tile_dict[0] = max(cache_tile_dict[0], tensor_sizes[0])
                    new_cache_tile_dict[1] = max(cache_tile_dict[1], tensor_sizes[1])
                    new_cache_tile_dict[2] = max(cache_tile_dict[2], capacities[1][2]*4)
                    dram_stationary_tensors = {2}
                if sum(new_cache_tile_dict.values()) > cache_size:
                    break
                else:
                    cache_tile_dict = new_cache_tile_dict
                    for tensor in dram_stationary_tensors:
                        dram_stationary_dict[tensor] = False
            
            # get W, I, O sizes
            cache_reads = [0, 0, 0]
            cache_updates = [0, 0, 0]
            for tensor in range(3):
                if not dram_stationary_dict[tensor]:
                    if reads[num_mem_lvls-1][tensor] > 0:
                        cache_reads[tensor] = reads[num_mem_lvls-1][tensor] - tensor_sizes[tensor]
                        reads[num_mem_lvls-1][tensor] = tensor_sizes[tensor]
            if not dram_stationary_dict[2]: # output updates
                cache_updates[2] = updates[num_mem_lvls-1][2] - tensor_sizes[2]
                updates[num_mem_lvls-1][2] = tensor_sizes[2]
        reads.append(cache_reads)
        updates.append(cache_updates)
        writes.append([0, 0, 0])

    return reads, updates, writes

def round_mapping(mapping: list[int], prob: Prob, round_perms = True, arch = None, inplace = False, round_down=False) -> list[int]:
    """
    Ensures tiling factors for each dimension multiply to total layer size in that dimension
    Also sets loop ordering based on relative value (lower value = more inner loop order)

    Loop ordering example:
        [P, Q, C, K] [-10, 5, 2, 9]
        K
            Q
                C
                    P
        [0, 2, 1, 3]

    TODO: write a version that uses imperfect factors?
    """
    # store new rounded mapping
    if inplace:
        new_mapping = mapping
        mapping = copy.deepcopy(mapping)
    else:
        new_mapping = copy.deepcopy(mapping)
    if isinstance(mapping, torch.Tensor):
        round_fn = torch.round
    elif isinstance(mapping, np.ndarray):
        round_fn = np.round
    else:
        round_fn = round
    # Get dim names and map to index, e.g. {R:0, S:1, P:2, Q:3, C:4, K:5, N:6}
    dim_idx_dict = prob.prob_name_idx_dict
    total_per_dim = {dim: 1 for dim in dim_idx_dict}
    num_dims = len(dim_idx_dict)
    num_mem_lvls = len(mapping) // num_dims // 3
    for mem_lvl in range(num_mem_lvls):
        start_idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3
        end_idx = start_idx + num_dims * 3
        mem_lvl_mapping = mapping[start_idx:end_idx]

        # spatial factors
        spatial_factors = mem_lvl_mapping[:num_dims]
        for dim, idx in dim_idx_dict.items():
            total_idx = start_idx + idx
            sf = max(1, round_fn(spatial_factors[idx]))
            if prob.prob[dim] % (total_per_dim[dim] * sf) == 0:
                pass
            else:
                # make sure it divides dim left
                # get the next factor
                dim_left = prob.prob[dim] // total_per_dim[dim]
                if round_down:
                    sf = utils.round_down_choices(spatial_factors[idx], utils.get_divisors(dim_left))
                else:
                    sf = utils.get_nearest_choice(spatial_factors[idx], utils.get_divisors(dim_left))
                # make sure it doesn't exceed prob dim
                max_factor = prob.prob[dim] // total_per_dim[dim]
                sf = min(max_factor, sf)
            total_per_dim[dim] *= sf
            new_mapping[total_idx] = sf

    for mem_lvl in range(num_mem_lvls):
        start_idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3
        end_idx = start_idx + num_dims * 3
        mem_lvl_mapping = mapping[start_idx:end_idx]
        # temporal factors
        temporal_factors = mem_lvl_mapping[num_dims:num_dims*2]
        for dim, idx in dim_idx_dict.items():
            total_idx = start_idx + num_dims + idx
            tf = max(1, round_fn(temporal_factors[idx]))
            if mem_lvl == num_mem_lvls - 1:
                tf = int(prob.prob[dim] / total_per_dim[dim])
            elif prob.prob[dim] % (total_per_dim[dim] * tf) == 0:
                pass
            else:
                # make sure it divides prob dim
                # get the next factor
                dim_left = prob.prob[dim] // total_per_dim[dim]
                if round_down:
                    tf = utils.round_down_choices(tf, utils.get_divisors(dim_left))
                else:
                    tf = utils.get_nearest_choice(tf, utils.get_divisors(dim_left))
                # make sure it doesn't exceed prob dim
                max_factor = prob.prob[dim] // total_per_dim[dim]
                tf = min(max_factor, tf)
            total_per_dim[dim] *= tf
            new_mapping[total_idx] = tf

        if round_perms:
            # permutation
            perms = mem_lvl_mapping[-num_dims:]
            perms_dict = {idx: perms[idx] for idx in range(len(perms))}
            sorted_perms = sorted(perms_dict.items(), key=lambda item: item[1])
            cur_perm = 1
            for idx, _ in sorted_perms:
                total_idx = start_idx + 2 * num_dims + idx
                new_mapping[total_idx] = cur_perm
                cur_perm += 1

    return new_mapping

def getProjection(self, mapping):
    """
    Code from https://github.com/kartik-hegde/mindmappings
    TODO: implement Mind Mappings method
    """
    # extract
    numHierarchy = self.arch['numHierarchy'] + 1
    numDims = len(self.problem['dimension_names'])
    ref_partition = self.references[-1]
    num_partitions = len(ref_partition[0][0])


    # 2. Extract
    tiling, loop_orders, partitions = mapping[self.parameters.TILING_IDX:self.parameters.LOOP_ORDER_IDX], \
                                        mapping[self.parameters.LOOP_ORDER_IDX:self.parameters.PARTITION_IDX], \
                                            mapping[self.parameters.PARTITION_IDX:]

    # Projection main procedure


    
    # #######  2. Loop Orders #######

        # We will simply sort the values and order the dimensions accordingly.
    loop_orders = [loop_orders[numDims*idx:numDims*(idx+1)] for idx in range(numHierarchy-1)]
    loop_orders = [list(np.argsort(loop_order)) for loop_order in loop_orders]
    loop_orders = [''.join([self.problem['dimension_names'][idx] for idx in loop_order]) for loop_order in loop_orders]

    # #######  3. Partitions #######

        # Here we will find the Euclidean distances to all the reference tiles per dimension, and choose the one
        # that has the minimum distance. --- strategy: Nearest Neighbor

    # First extract all partitions
    partitions = [partitions[idx*num_partitions:(idx+1)*num_partitions] for idx in range(numHierarchy-2)]
    # Get the partitions with minimum Euclidean distance
    partitions = [ref_partition[idx][np.argmin(cdist(ref_partition[idx], [partition,], metric='euclidean'))] for idx,partition in enumerate(partitions)]
    # Flatten it
    # partitions = [item for partition_sizes in partitions for item in partition_sizes]

    # #######  1. Tiling #######
        # Here we will find the Euclidean distances to all the reference tiles per dimension, and choose the one
        # that has the minimum distance. --- strategy: Nearest Neighbor

    # First, extract dimension wise tiling for memory hierarchies
    tiling = [tiling[numDims*h:numDims*(h+1)] for h in range(numHierarchy)]

    # Reference tiles
    ref_tiles = self.references[0]

    # Flatten
    tiling = [[tiling[h][idx] for h in range(numHierarchy)] for idx in range(numDims)]

    # Get all the tiles that form the minimum Euclidean distance
    distances = [np.reshape(cdist(ref_tiles[idx], [tiling[idx],], metric='euclidean'), (1,len(ref_tiles[idx]))) for idx in range(numDims)]
    indices = [np.argsort(dist[0]) for dist in distances]
    headers = [0]*numDims

    # Go in a loop till you find the closest valid tiling
    while True:
        tiling = [ref_tiles[idx][int(indices[idx][headers[idx]])] for idx in range(numDims)]
        tiling = [list(zip(*tiling))[h] for h in range(numHierarchy)]
        if(self.checkParallelValidity(tiling)):
            break
        else:
            target = np.argsort([distances[idx][0][indices[idx][headers[idx]]] for idx in range(numDims)])
            choice = 0
            validity = [idx for idx in range(numDims) if(headers[idx]+1 < len(ref_tiles[idx]))]
            if(not validity):
                return None
            while True:
                if(target[choice] in validity):
                    headers[target[choice]]+=1
                    break
                else:
                    choice += 1

    return [tiling, loop_orders, partitions]

if __name__ == "__main__":
    from dataset import DATASET_ROOT_PATH
    prob = Prob(DATASET_ROOT_PATH / "workloads" / "conv" / "conv_1.yaml")
    orig_mapping = [ 1,     1,     1,     1,     1,     1,
        1           ,0.99764484  ,0.9979975   ,1.0017812   ,1.1214931   ,1.0007585,
        1.2508408   ,1.          ,7.6709533   ,7.7267694   ,1.0131841   ,2.0185306,
        7.2102833   ,7.0868535   ,6.9948354   ,0.9999429   ,1.0001835   ,1.0005943,
        1.0212286   ,1.0001965   ,5.9301133   ,1.          ,1.0000083   ,0.9999989,
        1.0015384   ,1.9815457   ,1.0001965   ,2.086193    ,1.          ,7.,
        7.          ,7.          ,1.          ,7.          ,2.          ,7.,
        1.000074    ,1.000104    ,1.0007049   ,1.0001125   ,2.85024     ,1.0042933,
        1.          ,6.6206017   ,6.5457053   ,4.2763762   ,5.6711493   ,1.0009152,
        1.6708513   ,1.          ,2.0214717   ,2.995198    ,1.0009422   ,4.025372,
        7.014685    ,5.0629725   ,6.9768085   ,1.059749    ,1.0713959   ,2.135829,
        2.238605    ,1.0549359   ,2.2934415   ,1.          ,1.0001293   ,1.0001739,
        12.210412   ,3.8864524   ,1.0003077   ,1.0755686   ,1.          ,7.,
        7.          ,1.0013063   ,2.0059764   ,7.          ,7.          ,7.0349464 ]
    mapping = round_mapping(orig_mapping, prob, round_perms=False)
    print(mapping)
#     caps = capacity_from_mapping([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
# 1, 1, 2, 0, 6, 5, 4, 1, 3, 1, 1, 1, 1, 1, 16, 1, 1, 1, 4, 1, 1, 1, 1, 2, 4, 5, 6, 0, 1, 3, 1, 1, 1, 1, 32, 1, 1, 1, 1, 4, 28, 2, 4, 1, 0, 3, 2, 4, 5, 1,
# 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 3, 6, 0, 5, 1, 2], prob)
#     print(caps)
    # example_mapping = "L3[WIO] C8 Q7 K16 - L2[WI] Q2 C2 - L1[O] P7 K2 C4 K2X - L0[W] Q4 P8"
    # print(process_mapping(example_mapping, "cnn-layer"))
