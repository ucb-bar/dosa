from dataset import DATASET_ROOT_PATH
from dataset.common import utils

class Prob(object):
    """Problem space with layer dimension, stride and dilation defined.
    
    Attributes: 
        prob: A layer dimemsion dictionary. 
            R, S represent the weight filter width and height.
            P, Q represent the output feature map width and height.
            C represents the input channel size. 
            K represents the output channel size.
            N represents the batch size. 
            Wstride, Hstride represent the width and height dimension stride.
            Wdilation, Hdilation represent the width and height dimension dilation.
        prob_bound:  A 1d array with layer dimension value for R,S,P,Q,C,K,N
            e.g. [1,1,1,2,3,4,5]
        prob_factors:  A 2d array with all prime factors generated from each dimension
            e.g. [[1],[1],[1],[2],[3],[2,2],[5]] 
    """

    def __init__(self, prob_path):
        """Initialize the layer dimension from an input yaml file. 

            Example input yaml file format: 
                problem:
                  C: 3
                  Hdilation: 1
                  Hstride: 2
                  K: 64
                  N: 1
                  P: 112
                  Q: 112
                  R: 7
                  S: 7
                  Wdilation: 1
                  Wstride: 2
                  shape: cnn-layer


        Args: 
            prob_path: Path to the yaml file that defines the convolution layer dimensions. 
        """
        # defines the dimension index for 7 major loop bounds 
        self.prob_idx_name_dict = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
        self.prob_name_idx_dict = {v: k for k, v in self.prob_idx_name_dict.items()}

        self.prob_bound = [-1] * len(self.prob_name_idx_dict)
        self.prob_factors = []
        self.prob_divisors = []
        for i in range(len(self.prob_name_idx_dict)):
            self.prob_factors.append([])
            self.prob_divisors.append([])

        self.prob_levels = len(self.prob_idx_name_dict.items())

        store_prob = False
        if isinstance(prob_path, dict):
            prob_dict = prob_path
            store_prob = True
        else:
            self.path = prob_path.resolve()
            prob_dict = utils.parse_yaml(self.path)
        self.prob = prob_dict['problem']
        self.prob = self.prob.get('instance', self.prob) # if another level, index in
        self.shape = self.prob.get('shape', 'cnn-layer')

        for key, value in self.prob.items():
            if ('stride' in key or 'dilation' in key):
                continue
            if (key == 'shape'):
                continue
            prob_idx = self.prob_name_idx_dict[key]
            self.prob_bound[prob_idx] = value
            self.prob_factors[prob_idx] = utils.get_prime_factors(value)
            self.prob_divisors[prob_idx] = utils.get_divisors(value)

        if store_prob:
            generated_prob_dir = DATASET_ROOT_PATH / "workloads" / "generated_probs"
            utils.mkdir_p(generated_prob_dir)
            self.path = generated_prob_dir / (self.config_str() + ".yaml")
            if not self.path.is_file():
                utils.store_yaml(self.path, prob_path)

    def config_str(self):
        """Returnsthe key str name for representing a unique layer."""
        val_arr = []
        for value in self.prob_bound:
            val_arr.append(str(value))
        keys = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']
        val_arr.extend([str(self.prob[key]) for key in keys])
        val_str = "_".join(val_arr)
        return val_str

    def print(self):
        print(self.__dict__)
