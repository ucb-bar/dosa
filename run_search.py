import argparse
import pathlib

from dataset import DATASET_ROOT_PATH
from dataset.dse import mapping_driven_hw_search
from dataset.common import utils, logger

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Dataset Path',
                        required=True,
                        )
    parser.add_argument('-wl',
                        '--workload',
                        type=str,
                        help='<Required> Name of workload directory.',
                        required=True,
                        )
    parser.add_argument('--predictor',
                        type=str,
                        help='analytical, dnn, or both',
                        default='analytical',
                        )
    parser.add_argument('--plot_only',
                        action='store_true',
                        help='only plotting',
                        )
    parser.add_argument('--ordering',
                        type=str,
                        help='ordering',
                        default='shuffle',
                        )
    return parser

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir = f'{output_dir}'
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_driven_hw_search.search_network("gemmini", args.output_dir, args.workload, args.dataset_path, args.predictor, args.plot_only, args.ordering)
