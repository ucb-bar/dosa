import re
import subprocess
import pathlib
import time
import random
import string
import os
from collections.abc import Iterable

import numpy as np

from dataset import DATASET_ROOT_PATH
from dataset.common import utils, mapping_utils, logger
from dataset.workloads import Prob

class HardwareConfig():
    NAME = "DUMMY_NAME"
    BASE_ARCH_PATH = "DUMMY_ARCH_PATH"
    BASE_MAPSPACE_PATH = "DUMMY_MAPSPACE_PATH"

    def __init__(self, hw_config: str | Iterable | pathlib.Path, output_dir: pathlib.Path):
        """
        hw_config can be:
         - "random" (str), for random arch
         - a data structure definition for a hw config
         - a path to an existing YAML arch definition
        """
        self.hw_config = {}
        self.config_str = "DUMMY_CONFIG"
        self.config_dir = "DUMMY_CONFIG_DIR"
        self.arch_path = "DUMMY_ARCH_PATH"
        self.arch_dict = {}

        output_dir = pathlib.Path(output_dir).resolve()
        
        # Construct filename for new arch
        if hw_config is None or hw_config == "random":
            hw_config = self.gen_random_hw_config()
        elif pathlib.Path(str(hw_config)).resolve().is_file():
            hw_config = self.parse_arch_yaml(pathlib.Path(hw_config).resolve())
        self.hw_config = hw_config

        self.config_str = self.NAME
        for config in hw_config:
            self.config_str += "_" + str(config)

        # Define directory for all files of this config, and create that directory
        self.config_dir = output_dir / self.config_str
        pathlib.Path(self.config_dir).mkdir(parents=True, exist_ok=True)

        self.arch_path = self.config_dir / "arch.yaml"
        # arch_dict is a flattened representation used for the csv output
        self.arch_dict = self.gen_arch_yaml(hw_config)

    def parse_arch_yaml(self, arch_path: pathlib.Path) -> dict:
        logger.error("Running HardwareConfig's parse_arch_yaml(). Not defined for subclass.")
        raise NotImplementedError

    def get_config_str(self) -> str:
        return self.config_str

    def gen_random_hw_config(self):
        logger.error("Running HardwareConfig's gen_random_hw_config(). Not defined for subclass.")
        raise NotImplementedError

    def _create_timeloop_run_dir(self, layer_prob: Prob, custom_mapping: bool) -> pathlib.Path:
        output_path = self.config_dir / layer_prob.config_str()
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        timeline = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        this_run_output_dir = output_path / ("timeloop-" + timeline + "-" + randname)
        pathlib.Path(this_run_output_dir).mkdir(parents=True, exist_ok=True)
        return this_run_output_dir

    def flat_mapping_to_dict(self, prob_type: str, flat_mapping: list[int]) -> dict:
        """
        Flat mapping format: see mapping_utils.process_mapping()
        """
        # convert to int
        # flat_mapping = np.array(flat_mapping, dtype=int)
        flat_mapping = np.array(flat_mapping)
        mapspace_dict = utils.parse_yaml(self.BASE_MAPSPACE_PATH)
        if "mapspace_constraints" not in mapspace_dict:
            logger.error("Mapspace file %s should contain key 'mapspace_constraints'", 
                         self.BASE_MAPSPACE_PATH)
            return {}
        constraints_lst = mapspace_dict.get("mapspace_constraints")
        # collect names of mem lvls
        mapping_lst = [] # final mapping list (under key "mapping")
        targets = []
        for i, constraint in enumerate(constraints_lst):
            if constraint.get("type") == "bypass":
                mapping_lst.append(constraint)
                targets.append(constraint["target"])
        num_mem_lvls = len(targets)

        # check that we processed targets correctly
        dims = utils.get_prob_dims(prob_type)
        num_dims = len(dims)
        if len(flat_mapping) != num_mem_lvls * num_dims * 3:
            logger.error("Number of targets does not match prob type and mapping length.")
            logger.error("Mapping: %s, prob type: %s, targets: %s",
                         flat_mapping, prob_type, targets)
        for mem_lvl in range(num_mem_lvls):
            start_idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3
            end_idx = start_idx + num_dims * 3
            mem_lvl_mapping = flat_mapping[start_idx:end_idx]

            # target
            target = targets[mem_lvl]
            # spatial factors
            spatial_factors = mem_lvl_mapping[:num_dims]
            if self.NAME == "gemmini":
                for i in range(num_dims):
                    if not ((mem_lvl == 1 and dims[i] == "C") or (mem_lvl == 2 and dims[i] == "K")):
                        spatial_factors[i] = 1
            spatial_factor_str = " ".join([f"{dims[i]}={round(spatial_factors[i])}" for i in range(num_dims)])

            # only add spatial factors if there is a dim > 1
            spatial_add_cond = sum(spatial_factors) > len(spatial_factors)
            # in these cases we should add anyways
            spatial_add_cond = spatial_add_cond or (self.NAME == "gemmini" and (mem_lvl == 1 or mem_lvl == 2))
            if spatial_add_cond:
                mem_lvl_spatial_dict = {
                    "target": target,
                    "type": "spatial",
                    "factors": spatial_factor_str,
                    # TODO: add perm to handle X and Y dims?
                }
                mapping_lst.append(mem_lvl_spatial_dict)

            temporal_factors = mem_lvl_mapping[num_dims:num_dims*2]
            temporal_factor_str = " ".join([f"{dims[i]}={round(temporal_factors[i])}" for i in range(num_dims)])

            perms = mem_lvl_mapping[-num_dims:]
            num_factors_in_perm = 0
            perm_str_dict = {} # holds dim name in order corresponding to permutation
            for i in range(num_dims):
                if perms[i] < num_dims:
                    perm_str_dict[perms[i]] = dims[i]
                    num_factors_in_perm += 1
            perm_str_lst = [perm_str_dict[k] for k in sorted(perm_str_dict.keys())]
            perm_str = "".join(perm_str_lst[:num_factors_in_perm])

            mem_lvl_temporal_dict = {
                "target": target,
                "type": "temporal",
                "factors": temporal_factor_str,
                "permutation": perm_str,
            }

            mapping_lst.append(mem_lvl_temporal_dict)
        return {"mapping": mapping_lst}
    
    def run_mapping_from_dict(self, layer_prob: Prob, mapping_dict: dict, run_async: bool = False) -> dict:
        mapping = mapping_dict["mapping"]
        mapper_dict = {
            "algorithm": "exhaustive",
            "live-status": False,
            "num-threads": 8,
            "timeout": 0,
            "victory-condition": 1,
            "search-size": 1,
            "log-suboptimal": True,
            "diagnostics": True,
        }
        dict_to_mapper = {
            "mapper": mapper_dict,
            "mapspace_constraints": mapping
        }
        this_run_output_dir = self._create_timeloop_run_dir(layer_prob, True)
        mapping_path = this_run_output_dir / "custom_mapping.yaml"
        utils.store_yaml(mapping_path, dict_to_mapper)
        return self.run_mapping_from_file(layer_prob, mapping_path, this_run_output_dir, run_async=run_async)

    def run_mapping_from_file(self, layer_prob: Prob, mapping_path: pathlib.Path,
                              this_run_output_dir: pathlib.Path=None, run_async: bool=False) -> dict:
        """
        Runs a given mapping (Timeloop format) for a given layer

        Args:
            layer_path (pathlib.Path): path to layer definition.
            mapping_dict (dict): Timeloop format mapping definition. Includes
                tiling factors/permutations, as well as dataflow constraints.

        Returns:
            Dictionary with arch, prob, mapping and target features for this run.
        """
        # Create arch/prob output dir if it doesn't exist yet
        output_path = self.config_dir / layer_prob.config_str()
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        # Create output dir specific to this run, if needed
        if not this_run_output_dir:
            this_run_output_dir = self._create_timeloop_run_dir(layer_prob, True)
        # Run Timeloop and read result
        output_log_file = this_run_output_dir / "random.txt"
        with open(output_log_file, "w") as out_file:
            status_or_process = utils.run_timeloop_mapper(self.arch_path,
                                                  layer_prob.path,
                                                  mapping_path,
                                                  cwd=this_run_output_dir,
                                                  stdout=out_file,
                                                  stderr=subprocess.STDOUT,
                                                  run_async=run_async)
        if not status_or_process:
            return {}
        if not run_async:
            return self.read_run_mapping(layer_prob, mapping_path, output_log_file)
        else:
            read_bundle = (layer_prob, mapping_path, output_log_file)
            return status_or_process, read_bundle
    
    def read_run_mapping(self, layer_prob, mapping_path, output_log_file):
        rows = self.parse_random_output(output_log_file, layer_prob)
        if len(rows) == 1:
            logger.info("Successfully ran custom mapping %s", mapping_path)
        else:
            logger.error("Unable to run custom mapping %s on arch %s, layer %s", 
                         mapping_path, self.arch_path, layer_prob.config_str())
            return {}
        return rows[0]

    def run_random_mappings(self, layer_prob: Prob, num_mappings: int, exist: bool = False, return_min_fn=None) -> list[dict]:
        """
        Uses the Timeloop random mapper to collect perf numbers for random mappings on a given layer
        """
        logger.info("Running %d random mappings for layer: %s; on arch %s, exist=%s",
                    num_mappings, layer_prob.config_str(), self.get_config_str(), exist)
        rows = []  # return value
        mapspace_path = self.BASE_MAPSPACE_PATH
        output_path = self.config_dir / (layer_prob.config_str())
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        mapspace_dict = utils.parse_yaml(mapspace_path)
        mapspace_dict["mapper"]["num-threads"] = min(num_mappings, 10)
        mapspace_dict["mapper"]["search-size"] = num_mappings
        # TODO: figure out whether 10x mappings is reasonable timeout?
        mapspace_dict["mapper"]["timeout"]     = max(num_mappings * 10, 1000)
        if self.NAME == "gemmini":
            pe_dim = int(self.hw_config[0])
            for entry in mapspace_dict["mapspace_constraints"]:
                if entry["target"] == "Accumulator" and entry.get("type") == "spatial":
                    # Accumulator spatial constraints
                    entry["factors"] = f"R=1 S=1 P=1 Q=1 C<={pe_dim} K=1 N=1"
                elif entry["target"] == "Scratchpad" and entry.get("type") == "spatial":
                    # Scratchpad spatial constraints
                    entry["factors"] = f"R=1 S=1 P=1 Q=1 N=1 C=1 K<={pe_dim}"
        new_mapspace_path = self.config_dir / "mapspace.yaml"
        utils.store_yaml(new_mapspace_path, mapspace_dict)

        output_log_file = output_path / "random.txt"
        if not exist:
            with open(output_log_file, "w") as out_file:
                success = utils.run_timeloop_mapper(self.arch_path,
                                                    layer_prob.path,
                                                    new_mapspace_path,
                                                    cwd=output_path,
                                                    stdout=out_file,
                                                    stderr=subprocess.STDOUT)
                if not success:
                    return []
        rows = self.parse_random_output(output_log_file, layer_prob)
        if len(rows) == num_mappings:
            logger.info("Successfully ran %d random mappings", len(rows))
        else:
            logger.warning("Ran %d random mappings, not the %d requested. See %s for details.",
                        len(rows), num_mappings, output_log_file)
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

    def run_exhaustive_mappings(self, layer_prob: Prob, exist: bool = False, return_min_fn=None) -> list[dict]:
        """
        Uses the Timeloop random mapper to collect perf numbers for random mappings on a given layer
        """
        logger.info("Running exhaustive mappings for layer: %s; on arch %s, exist=%s",
                    layer_prob.config_str(), self.get_config_str(), exist)
        rows = []  # return value
        mapspace_path = self.BASE_MAPSPACE_PATH
        output_path = self.config_dir / (layer_prob.config_str())
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        mapspace_dict = utils.parse_yaml(mapspace_path)
        mapspace_dict["mapper"]["algorithm"] = "linear-pruned"
        mapspace_dict["mapper"]["num-threads"] = 16
        mapspace_dict["mapper"]["timeout"] = 0
        mapspace_dict["mapper"].pop("search-size")
        # TODO: figure out whether 10x mappings is reasonable timeout?
        # mapspace_dict["mapper"]["timeout"]     = max(num_mappings * 10, 1000)
        if self.NAME == "gemmini":
            pe_dim = int(self.hw_config[0])
            for entry in mapspace_dict["mapspace_constraints"]:
                if entry["target"] == "Accumulator" and entry.get("type") == "spatial":
                    # Accumulator spatial constraints
                    entry["factors"] = f"R=1 S=1 P=1 Q=1 C<={pe_dim} K=1 N=1"
                elif entry["target"] == "Scratchpad" and entry.get("type") == "spatial":
                    # Scratchpad spatial constraints
                    entry["factors"] = f"R=1 S=1 P=1 Q=1 N=1 C=1 K<={pe_dim}"
        new_mapspace_path = self.config_dir / "mapspace.yaml"
        utils.store_yaml(new_mapspace_path, mapspace_dict)

        output_log_file = output_path / "random.txt"
        if not exist:
            with open(output_log_file, "w") as out_file:
                success = utils.run_timeloop_mapper(self.arch_path,
                                                    layer_prob.path,
                                                    new_mapspace_path,
                                                    cwd=output_path,
                                                    stdout=out_file,
                                                    stderr=subprocess.STDOUT)
                if not success:
                    return []
        rows = self.parse_random_output(output_log_file, layer_prob)
        if not rows:
            logger.warning("Failed to exhaustively run mappings")
        else:
            logger.info("Exhaustively ran %d random mappings", len(rows))
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

    def run_cosa(self, layer_prob: Prob, exist: bool = False, run_mapping: bool = True, run_async: bool = False) -> list[dict]:
        """
        Generates CoSA mapping for given layer
        """
        logger.info("Running CoSA mapper for layer: %s; on arch %s, exist=%s",
                    layer_prob.path, self.get_config_str(), exist)
        rows = []  # return value
        mapspace_path = self.BASE_MAPSPACE_PATH
        output_path = self.config_dir / layer_prob.config_str()
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # need to use cosa specific arch without meshX attribute
        prob_path = pathlib.Path(layer_prob.path).resolve()
        mapspace_path= pathlib.Path(mapspace_path).resolve()
        output_dir = pathlib.Path(output_path).resolve()
        new_mapspace_path = output_path / "mapspace.yaml"
        output_mapper_yaml_path = pathlib.Path(new_mapspace_path).resolve()
        
        # must specify COSA_DIR
        cosa_dir = DATASET_ROOT_PATH.parent / "mapper" / "cosa-gemmini"
        os.environ["COSA_DIR"] = str(cosa_dir) # needed to run cosa
        if not run_async:
            utils.run_cosa(self.arch_path, prob_path, mapspace_path, output_dir, output_mapper_yaml_path, cwd=cosa_dir, stdout=None, stderr=None)
            return self.collect_cosa_result(layer_prob, exist=exist, run_mapping=run_mapping)
        else:
            proc = utils.run_cosa_async(self.arch_path, prob_path, mapspace_path, output_dir, output_mapper_yaml_path, cwd=cosa_dir, stdout=None, stderr=None)
            return proc

    def collect_cosa_result(self, layer_prob, exist=False, run_mapping=True):
        # copied from run_cosa, make sure paths match
        prob_path = pathlib.Path(layer_prob.path).resolve()
        mapspace_path= pathlib.Path(self.BASE_MAPSPACE_PATH).resolve()
        output_path = self.config_dir / layer_prob.config_str()
        new_mapspace_path = output_path / "mapspace.yaml"
        output_mapper_yaml_path = pathlib.Path(new_mapspace_path).resolve()

        # merge mapper bypass and cosa generated mapping constraints
        mapspace_path = self.BASE_MAPSPACE_PATH
        mapspace_setup = utils.parse_yaml(mapspace_path)
        # JENNY: Need to unify the mapspace format
        mapspace_dict = utils.parse_yaml(output_mapper_yaml_path)
        if not run_mapping:
            flat_mapping = self.parse_mapping(mapspace_dict)
            return flat_mapping

        mapspace_dict['mapper'] = mapspace_setup['mapper']
        mapspace_dict["mapper"]["search-size"] = 1
        mapspace_dict["mapper"]["num-threads"] = 8
        mapspace_dict["mapper"]["sync-interval"] = 1

        utils.store_yaml(output_mapper_yaml_path, mapspace_dict)

        prob_path = pathlib.Path(layer_prob.path).resolve()
        output_log_file = output_path / "cosa.txt"
        if not exist:
            with open(output_log_file, "w") as out_file:
                success = utils.run_timeloop_mapper(self.arch_path,
                                                    prob_path,
                                                    new_mapspace_path,
                                                    cwd=output_path,
                                                    stdout=out_file,
                                                    stderr=subprocess.STDOUT)
                if not success:
                    return []
        # why parse log but not the stats file
        rows = [self.parse_random_output(output_log_file, layer_prob)[0]]
        if len(rows) == 1:
           logger.info("Successfully ran CoSA mapping")
        else:
            logger.warning("Ran %d CoSA mappings, not the %d requested. See %s for details.",
                        len(rows), 1, output_log_file)
        return rows

    def parse_mapping(self, mapping_dict: dict):
        if "mapping" in mapping_dict:
            mapping = mapping_dict["mapping"]
        elif "architecture_constraints" in mapping_dict or "mapspace_constraints" in mapping_dict:
            mapping = mapping_dict.get("architecture_constraints", mapping_dict.get("mapspace_constraints"))
            mapping = [entry for entry in mapping if "bypass" not in entry]
        num_mem_lvls = len(self.MEM_LVLS)

        temporal_strings = [""] * num_mem_lvls
        spatial_strings = [""] * num_mem_lvls
        for entry in mapping:
            factor_type = entry["type"]
            if factor_type != "bypass":
                target = entry["target"]
                factors_lst = entry["factors"].split()
                factors = {}
                for item in factors_lst:
                    k, v = item.split("=")
                    factors[k] = int(v)
                perm = entry.get("permutation", "")
                if len(perm) < 7:
                    for dim in "RSPQCKN":
                        if dim not in perm:
                            perm += dim
                mem_lvl = list(self.MEM_LVLS.keys()).index(target)
                for dim in reversed(perm):
                    if factor_type == "temporal":
                        if factors[dim] > 1:
                            temporal_strings[mem_lvl] += f" {dim}{factors[dim]}"
                    if factor_type == "spatial":
                        if factors[dim] > 1:
                            spatial_strings[mem_lvl] += f" {dim}{factors[dim]}X"

        for mem_lvl in range(num_mem_lvls):
            if temporal_strings[mem_lvl] == "" and spatial_strings[mem_lvl] == "":
                temporal_strings[mem_lvl] = " N1"

        strings = []
        for mem_lvl in range(num_mem_lvls):
            holds = list(self.MEM_LVLS.values())[mem_lvl]
            strings.append(f"L{mem_lvl}[{holds}]" + temporal_strings[mem_lvl] + spatial_strings[mem_lvl])

        mapping_str = " - ".join(reversed(strings))
        logger.debug("mapping_str: %s", mapping_str)
        flat_mapping = mapping_utils.process_mapping(mapping_str, "cnn-layer")
        return flat_mapping

    def parse_random_output(self, output_log_file: pathlib.Path, layer_prob: Prob) -> list[dict]:
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

        arch_dict_flattened = {}
        for key, val in self.arch_dict.items():
            arch_dict_flattened[f"arch.{key}"] = val
        arch_dict_flattened["arch.name"] = self.NAME

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

                row.update(arch_dict_flattened)
                row.update(prob_dict_flattened)

                # if cycle < min_cycle:
                #     min_cycle = cycle

                rows.append(row)
        return rows
