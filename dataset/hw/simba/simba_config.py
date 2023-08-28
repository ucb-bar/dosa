import pathlib
import re
import copy
import random
from collections.abc import Iterable

from dataset.common import utils, logger
from dataset.hw import HardwareConfig

SIMBA_DIR = pathlib.Path(__file__).parent.resolve()

class SimbaConfig(HardwareConfig):
    NAME = "simba"
    BASE_ARCH_PATH = pathlib.Path(f"{SIMBA_DIR}/arch/arch.yaml").resolve()
    BASE_MAPSPACE_PATH = pathlib.Path(f"{SIMBA_DIR}/mapspace/mapspace_random.yaml").resolve()
    MEM_LVLS = {
        "Registers": "W",
        "AccumulationBuffer": "O",
        "WeightBuffer": "W", 
        "InputBuffer": "I", 
        "GlobalBuffer": "IO",
        "DRAM": "WIO",
    }

    def gen_random_hw_config(self) -> list[int]:
        """
        mac_num is number of MACs per vector unit (8 vector units per PE)
        """
        # pe_num,mac_num, \
        #     mem1_depth,mem1_blocksize,mem1_ports,mem1_banks, \
        #     mem2_depth,mem2_blocksize,mem2_ports,mem2_banks, \
        #     mem3_depth,mem3_blocksize,mem3_ports,mem3_banks, \
        #     mem4_depth,mem4_blocksize,mem4_ports,mem4_banks = hw_config
        # default_hw_config = [16, 8*128,
        #     64, 1, 2, 1,
        #     2048, 8, 2, 1,
        #     64, 8, 2, 1,
        #     8192, 8, 2, 1
        # ]
        max_bounds = [16*4, 8*4,
            16*4, 8*4, 2, 2,
            512*4, 8*4, 2, 8,
            1024*4, 8*4, 2, 8,
            8192*4, 8*4, 2, 8
        ]
        min_bounds = [16//4, 8//4,
            16//4, 1, 1, 1,
            512//4, 1, 1, 1,
            1024//4, 1, 1, 1,
            8192//4, 1, 1, 1
        ]
        options_per_config = []
        for i in range(len(max_bounds)):
            options = []
            option = max_bounds[i]
            while option >= min_bounds[i]:
                options.append(option)
                option = option // 2
            options_per_config.append(options)
        hw_config = []
        for options in options_per_config:
            hw_config.append(random.choice(options))
        logger.debug("Generated random simba config: %s", hw_config)
        return hw_config

    def gen_arch_yaml(self, hw_config: Iterable[int]) -> dict:
        logger.debug("Generating arch YAML for %s", self.get_config_str())

        # arch_dict is flattened arch definition, to be returned by this fn
        arch_dict = {}

        # Get base arch dictionary
        base_arch = utils.parse_yaml(SimbaConfig.BASE_ARCH_PATH)

        new_arch = copy.deepcopy(base_arch)
        # base_arch_dict = base_arch["architecture"]["subtree"][0]["subtree"][0]
        chip_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]

        ##### Calculate SRAM attributes
        BUFFER_ARRAY_LEN = 8

        pe_num,mac_num, \
            mem1_depth,mem1_blocksize,mem1_ports,mem1_banks, \
            mem2_depth,mem2_blocksize,mem2_ports,mem2_banks, \
            mem3_depth,mem3_blocksize,mem3_ports,mem3_banks, \
            mem4_depth,mem4_blocksize,mem4_ports,mem4_banks = hw_config

        buf_attributes = {
            0: {"depth": 1, "blocksize": 1, "ports": 2, "banks": 8}, # registers
            1: {"depth": mem1_depth, "blocksize": mem1_blocksize, "ports": mem1_ports, "banks": mem1_banks}, # acc
            2: {"depth": mem2_depth, "blocksize": mem2_blocksize, "ports": mem2_ports, "banks": mem2_banks}, # weight
            3: {"depth": mem3_depth, "blocksize": mem3_blocksize, "ports": mem3_ports, "banks": mem3_banks}, # input
            4: {"depth": mem4_depth, "blocksize": mem4_blocksize, "ports": mem4_ports, "banks": mem4_banks}, # global
        }

        arch_dict["pe"] = pe_num
        arch_dict["mac"] = mac_num
        mesh_x = pe_num # TODO: don't just use num PEs as placeholder?
        arch_dict["meshX"] = mesh_x

        # Set values in arch YAML
        logger.debug("Setting pe_num %d, buf_attributes %s", pe_num, buf_attributes)

        arch_invalid = False
        chip_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]

        # Set meshX and instances for MACs and corresponding registers
        logger.error(chip_dict["subtree"][0])
        new_arith = chip_dict["subtree"][0]["subtree"][0]["subtree"][0]
        registers = new_arith["local"][0]["attributes"]
        macs = new_arith["local"][1]["attributes"]
        new_arith["name"] = f"VectorArray[0..{mac_num - 1}]"

        macs["meshX"] = mesh_x
        total_mac_num = pe_num * BUFFER_ARRAY_LEN * mac_num
        macs["instances"] = total_mac_num
        registers["cluster-size"] = pe_num * BUFFER_ARRAY_LEN

        pe_dict = chip_dict["subtree"][0]
        buf_arr_dict = pe_dict["subtree"][0]
        pe_dict["name"] = f"PE[0..{pe_num - 1}]"
        buf_arr_dict["name"] = f"BufferArray[0..{BUFFER_ARRAY_LEN - 1}]"

        # Set buffer attributes
        # accbuf (lvl 1), weightbuf (lvl 2), inputbuf (lvl 3), globalbuf (lvl 4)
        mem_lvl_to_attrs = {
            0: registers,
            1: buf_arr_dict["local"][1]["attributes"],
            2: buf_arr_dict["local"][0]["attributes"],
            3: pe_dict["local"][0]["attributes"],
            4: chip_dict["local"][0]["attributes"],
        }

        mem_lvl_to_instances = {
            0: total_mac_num,
            1: pe_num * BUFFER_ARRAY_LEN,
            2: pe_num * BUFFER_ARRAY_LEN,
            3: pe_num,
            4: 1,
        }

        for mem_lvl in range(0, 5): # mem_lvl is mem lvl
            attrs = mem_lvl_to_attrs[mem_lvl]
            if "meshX" in attrs:
                attrs["meshX"] = mesh_x
                # Check whether meshX divides num instances of all buffers
                # if attrs["instances"] % attrs["meshX"] != 0:
                #     logger.error("Arch invalid. Instances: %s, meshX: %s",
                #         attrs["instances"],
                #         attrs["meshX"])
                #     arch_invalid = True

            depth = buf_attributes[mem_lvl]["depth"]
            blocksize = buf_attributes[mem_lvl]["blocksize"]
            cluster_size = attrs.get("cluster-size", 1) #  get orig cluster size
            word_bits = attrs["word-bits"] # keep orig word-bits
            attrs["block-size"] = blocksize
            attrs["depth"] = depth
            attrs["entries"] = depth * blocksize
            attrs["width"] = blocksize * word_bits * cluster_size
            attrs["num-ports"] = buf_attributes[mem_lvl]["ports"]
            attrs["num-banks"] = buf_attributes[mem_lvl]["banks"]
            attrs["instances"] = mem_lvl_to_instances[mem_lvl]

            new_attributes = attrs
            for key in new_attributes:
                arch_dict[f"mem{mem_lvl}_{key}"] = new_attributes[key]

            # # Check whether SRAM size is at least 64
            # entries = attrs["entries"]
            # banks = attrs["num-banks"]
            # if (entries // banks) < 64:
            #     print("Arch invalid:")
            #     print("Mem lvl", mem_lvl, "Entries:", entries, "Banks:", banks)
            #     arch_invalid = True
        
        if arch_invalid:
            logger.error("Arch invalid: %s", self.get_config_str())
            return

        # Save new arch
        utils.store_yaml(self.arch_path, new_arch)
        logger.info("Stored new arch YAML to %s", self.arch_path)

        return arch_dict

    def parse_arch_yaml(self, arch_path: pathlib.Path) -> list:
        """
        Not tested. TODO implement/debug.
        """
        base_arch_str = arch_path.name
        m = re.search("(\S+).yaml", base_arch_str)
        if not m:
            logger.error("Wrong config string format.")
            raise
        config_str = m.group(1)
        logger.debug(config_str)

        new_arch = utils.parse_yaml(arch_path)

        base_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        base_meshX_str =  base_arch_dict["subtree"][0]["name"]
        m = re.search("PE\[0..(\S+)\]", base_meshX_str)
        if not m:
            logger.error("Wrong mesh-X specification.")
            raise
        base_meshX = int(m.group(1)) + 1

        base_arith = base_arch_dict["subtree"][0]["local"][4]["attributes"]
        base_storage = base_arch_dict["subtree"][0]["local"]
        data_entry = [str(base_meshX), str(base_arith["instances"])]
        
        for i in reversed(range(1, 4)): 
            data_entry.extend([str(base_storage[i]["attributes"]["instances"]),str(base_storage[i]["attributes"]["entries"])])
        base_gb_dict = base_arch_dict["local"][0]
        data_entry.extend([str(base_gb_dict["attributes"]["instances"]), str(base_gb_dict["attributes"]["entries"])])

        return data_entry

        # data_entry = [str(cycle), str(energy)] + [str(area), str(edp), str(adp)]  + data_entry
        # data_entry = [str(cycle), str(energy)] + data_entry
        # data.append((config_str, data_entry))
        # return data
