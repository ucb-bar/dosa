# __init__.py
from dataset.common import logger
from .hardware_config import HardwareConfig
from .gemmini.gemmini_config import GemminiConfig
from .simba.simba_config import SimbaConfig

def init_hw_config(arch_name: str, *args) -> HardwareConfig:
    if arch_name == "gemmini":
        return GemminiConfig(*args)
    elif arch_name == "simba":
        return SimbaConfig(*args)
    else:
        logger.error("Arch %s not a supported subclass of HardwareConfig; see %s", arch_name, __file__)
