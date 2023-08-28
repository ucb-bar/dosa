from distutils.command.config import config
import pathlib

from dataset.common import utils, logger
from dataset.workloads import Prob

WORKLOAD_DIR = pathlib.Path(__file__).parent.resolve()

def is_unique_prob(prob, probs):
    for exist_prob in probs:
        if prob == exist_prob:
            return False
    return True

def get_unique_layers(model_name: str):
    """
    Generate a yaml describing the layer counts of a model, e.g. resnet50

    The directory must have the following structure:
    The layers come from pytorch model zoo # TODO: find script to generate these

    workloads
        |-- <model_name>
            |-- layers.yaml (contains all layer names)
            |-- <layer_1_name>.yaml
            |-- <layer_2_name>.yaml
            |-- ...
    """    
    model_dir = WORKLOAD_DIR / (model_name)

    layer_def_path = model_dir / 'layers.yaml'
    layers = utils.parse_yaml(layer_def_path)

    layer_dicts = []
    unique_layers= []
    layer_count = {}
    for layer in layers:
        prob_path = model_dir / (layer + '.yaml') 
        prob = Prob(prob_path)
        config_str = prob.config_str()
        if config_str not in layer_count:
            layer_count[config_str] = {"name": layer, "count": 1}
        else:
            layer_count[config_str]["count"] = layer_count[config_str]["count"] + 1
        
        if is_unique_prob(prob.prob, layer_dicts): 
            layer_dicts.append(prob.prob)
            unique_layers.append(layer)

    unique_layers_path = model_dir / 'unique_layers.yaml'
    layer_count_path = model_dir / 'layer_count.yaml'
    utils.store_yaml(unique_layers_path, unique_layers)
    utils.store_yaml(layer_count_path, layer_count)

    print(layer_count)
    total = sum((c["count"] for c in layer_count.values()))
    logger.info(model_name + f" layers counted : {total}")

if __name__ == "__main__":
    get_unique_layers("unet")