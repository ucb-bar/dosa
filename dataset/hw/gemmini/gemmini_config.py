import pathlib
import re
import copy
import random
from collections.abc import Iterable
import math

from dataset.common import utils, logger
from dataset.hw import HardwareConfig

GEMMINI_DIR = pathlib.Path(__file__).parent.resolve()

class GemminiConfig(HardwareConfig):
    """
    Gemmini arch class.

    Instances of this class represent different hardware configs of Gemmini.
    hw_config format:
        [pe_dim, sp_size (KB), acc_size (KB)]
    """
    NAME = "gemmini"
    BASE_ARCH_PATH = pathlib.Path(f"{GEMMINI_DIR}/arch/arch.yaml").resolve()
    BASE_MAPSPACE_PATH = pathlib.Path(f"{GEMMINI_DIR}/mapspace/mapspace_real_gemmini.yaml").resolve()
    # BASE_MAPSPACE_PATH = pathlib.Path(f"{GEMMINI_DIR}/mapspace/mapspace_random.yaml").resolve()
    BASE_PE = 16
    BASE_SP_SIZE = 128 # in KB
    BASE_ACC_SIZE = 32 # in KB
    MEM_LVLS = {
        "Registers": "W",
        "Accumulator": "O",
        "Scratchpad": "WI",
        "DRAM": "WIO",
    }

    def gen_random_hw_config(self):
        pe_multiplier = random.choice([0.125, 0.25, 0.5, 0.75, 1, 2, 3, 4, 6, 8, 12, 16])
        # pe_multiplier = random.choice([0.125, 0.25, 0.5, 1, 2, 4])
        # pe_multiplier = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]) # high fidelity for training 
        # pe_multiplier = random.choice([0.5, 1, 2, 4, 8, 16, 24, 32])
        buf_multipliers = [
            # random.randrange(1, 129, 1) / 16, # effectively randrange(0.0625, 8.0625, 0.0625)
            # random.randrange(1, 129, 1) / 16,
            # random.randrange(1, 129, 1) / 4,
            # random.randrange(1, 129, 1) / 4,
            # random.choice([1/8, 1/4, 3/8, 1/2, 3/4, 1, 2, 3, 4]), # small starts for search_gd
            # random.choice([1/8, 1/4, 3/8, 1/2, 3/4, 1, 2, 3, 4, 6, 8]),
            # random.choice([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]), # high fidelity for training
            # random.choice([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
            # random.randrange(1, 2**6+1, 1), # 100000arch dataset for training area model
            # random.randrange(1, 2**6+1, 1),
        ]
        hw_config = [
            # int(GemminiConfig.BASE_PE * pe_multiplier),
            # int(GemminiConfig.BASE_SP_SIZE * buf_multipliers[0]),
            # int(GemminiConfig.BASE_ACC_SIZE * buf_multipliers[1]),
            2 ** random.randrange(1, 8, 1),
            2 ** random.randrange(0, 12, 1),
            2 ** random.randrange(0, 12, 1),
        ]
        # hw_config = [
        #     int(pe_multiplier),
        #     buf_multipliers[0],
        #     buf_multipliers[1],
        # ]
        return hw_config

    def gen_arch_yaml(self, hw_config):
        logger.debug("Generating arch YAML for %s", self.get_config_str())

        # arch_dict is flattened arch definition, to be returned by this fn
        arch_dict = {}

        # Get base arch dictionary
        base_arch = utils.parse_yaml(GemminiConfig.BASE_ARCH_PATH)

        new_arch = copy.deepcopy(base_arch)
        # base_arch_dict = base_arch["architecture"]["subtree"][0]["subtree"][0]
        chip_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]

        pe_dim = int(hw_config[0]) # of PEs
        arch_dict["pe"] = pe_dim * pe_dim
        arch_dict["mac"] = 1
        arch_dict["meshX"] = pe_dim

        # Scratchpad and accumulator sizes in KB
        sp_size = hw_config[1] * 1024 # in B
        acc_size = hw_config[2] * 1024 # in B

        # index 0: scratchpad
        # index 1: accumulator
        buf_attributes = []

        ##### Calculate SRAM attributes
        sp_word_bits = 8
        sp_width = int(pe_dim * sp_word_bits) # of bits
        sp_depth = math.ceil(sp_size / (sp_width // 8)) # of rows

        # sp_read_bw = 16 # total
        # sp_write_bw = 16
        # acc_read_bw = 16 # total
        # acc_write_bw = 16
        dram_shared_bw = 8
        if len(hw_config) == 8:
            sp_read_bw = round(hw_config[3], 3)
            sp_write_bw = round(hw_config[4], 3)
            acc_read_bw = round(hw_config[5], 3)
            acc_write_bw = round(hw_config[6], 3)
            dram_shared_bw = round(hw_config[7], 3)

        buf_attributes.append({ # Scratchpad attributes
            "depth": sp_depth,
            "width": sp_width,
            "word-bits": sp_word_bits,
            "blocksize": pe_dim,
            "instances": 1,
            "shared_bandwidth": pe_dim * 2,
            # "read_bandwidth": sp_read_bw,
            # "write_bandwidth": sp_write_bw,
        })

        acc_word_bits = 32
        acc_width = int(pe_dim * acc_word_bits) # of bits
        acc_depth = math.ceil(acc_size / (acc_width // 8)) # of rows

        # Accumulator is instantiated over pe_dim number of copies;
        # the purpose of this is to prevent overflowing the Gemmini accumulator,
        # since it stores outputs per row in the accumulator
        chip_dict["subtree"][0]["name"] = f"PECols[0..{pe_dim-1}]"
        buf_attributes.append({ # Accumulator attributes
            "depth": acc_depth,
            "width": acc_width // pe_dim,
            "word-bits": acc_word_bits,
            "blocksize": 1,
            "instances": pe_dim,
            "meshX": pe_dim,
            "shared_bandwidth": 2,
            # "read_bandwidth": acc_read_bw / pe_dim,
            # "write_bandwidth": acc_write_bw / pe_dim,
        })

        ##### Set values in arch YAML

        logger.debug("Setting pe_dim %d, sp_size %d B, acc_size %d B", pe_dim, sp_size, acc_size)
        # base_meshX_str = base_arch_dict["subtree"][0]["name"]
        # m = re.search("PE\[0..(\S+)\]", base_meshX_str)
        # if not m:
        #     raise ValueError("Wrong mesh-X specification.")
        # base_meshX = int(m.group(1)) + 1
        # new_meshX = int(base_meshX * pe_multiplier) - 1 
        # chip_dict["subtree"][0]["name"] = f"PE[0..{new_meshX}]" 

        # Set DRAM BW
        dram_attrs = new_arch["architecture"]["subtree"][0]["local"][0]["attributes"]
        dram_attrs["shared_bandwidth"] = dram_shared_bw

        # Set arith and registers to match PEs
        pe_dict = chip_dict["subtree"][0]["subtree"][0]
        pe_dict["name"] = f"PERows[0..{pe_dim-1}]"

        new_arith = pe_dict["local"][1]["attributes"]
        # new_arith["meshX"] = pe_dim

        new_reg = pe_dict["local"][0]["attributes"]
        # new_reg["meshX"] = pe_dim
        new_reg["instances"] = pe_dim * pe_dim

        # Set buffer attributes
        # acc (lvl 1), sp (lvl 2)
        mem_lvl_to_attrs = {
            1: chip_dict["subtree"][0]["local"][0]["attributes"],
            2: chip_dict["local"][0]["attributes"],
        }

        arch_invalid = False
        for mem_lvl in range(1, 3): 
            # mem_lvl is memory level
            # 1: Accumulator
            # 2: Scratchpad

            # i is index in buf_attributes list
            # 0: Scratchpad
            # 1: Accumulator
            i = 2 - mem_lvl

            attrs = mem_lvl_to_attrs[mem_lvl]

            depth = buf_attributes[i]["depth"]
            word_bits = buf_attributes[i]["word-bits"]
            blocksize = buf_attributes[i]["blocksize"]
            attrs["entries"] = depth * blocksize
            attrs["depth"] = depth
            attrs["width"] = blocksize * word_bits
            attrs["instances"] = buf_attributes[i]["instances"]
            if "shared_bandwidth" in buf_attributes[i]:
                attrs["shared_bandwidth"] = buf_attributes[i]["shared_bandwidth"]
            # attrs["n_rdwr_ports"] = buf_attributes[mem_lvl]["ports"]
            # attrs["n_banks"] = buf_attributes[mem_lvl]["banks"]
            if "meshX" in attrs and "meshX" in buf_attributes[i]:
                attrs["meshX"] = buf_attributes[i]["meshX"]
                # Check whether meshX divides num instances of all buffers
                if attrs["instances"] % attrs["meshX"] != 0:
                    logger.warning("Arch invalid")
                    logger.warning("Instances: %s", attrs["instances"])
                    logger.warning("meshX: %s", attrs["meshX"])
                    arch_invalid = True

            for key in attrs:
                arch_dict[f"mem{mem_lvl}_{key}"] = attrs[key]

        # # global buffer (simulated L2 cache)
        # base_gb_dict = base_arch_dict["local"][0]
        # new_gb_dict = chip_dict["local"][0]
        # new_gb_dict["attributes"]["entries"] = int(base_gb_dict["attributes"]["entries"] * buf_multipliers_perm[3])
        
        if arch_invalid:
            logger.error("Arch invalid: %s", self.get_config_str())
            return

        # Save new arch
        utils.store_yaml(self.arch_path, new_arch)
        logger.debug("Stored new arch YAML to %s", self.arch_path)

        return arch_dict

    def parse_arch_yaml(self, arch_path):
        """
        Returns HW config
        """
        base_arch_str = arch_path.name
        m = re.search("(\S+).yaml", base_arch_str)
        if not m:
            logger.error("Wrong config string format.")
            raise
        config_str = m.group(1)
        logger.debug(config_str)

        new_arch = utils.parse_yaml(arch_path)

        chip_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        pe_width_str = chip_dict["subtree"][0]["name"]
        pe_height_str = chip_dict["subtree"][0]["subtree"][0]["name"]
        m = re.search("PECols\[0..(\S+)\]", pe_width_str)
        if not m:
            logger.error("Couldn't find PE width.")
            raise
        pe_width = int(m.group(1)) + 1
        m = re.search("PERows\[0..(\S+)\]", pe_height_str)
        if not m:
            logger.error("Couldn't find PE height.")
            raise
        pe_height = int(m.group(1)) + 1
        num_pe = pe_width * pe_height
        hw_config = [int(num_pe ** 0.5)]

        sp_attr = chip_dict["local"][0]["attributes"]
        depth = int(sp_attr["depth"])
        width = int(sp_attr["width"])
        mem_size = depth * width / 8 / 1024 # bits -> KB
        hw_config.append(mem_size)

        acc_attr = chip_dict["subtree"][0]["local"][0]["attributes"]
        depth = int(acc_attr["depth"])
        width = int(acc_attr["width"]) * pe_width
        mem_size = depth * width / 8 / 1024 # bits -> KB
        hw_config.append(mem_size)

        return hw_config

        # data_entry = [str(cycle), str(energy)] + [str(area), str(edp), str(adp)]  + data_entry
        # data_entry = [str(cycle), str(energy)] + data_entry
        # data.append((config_str, data_entry))
        # return data
