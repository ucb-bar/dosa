import pathlib
import re
import time
from typing import List
import math
import multiprocessing
import os
import copy

# try:
#     import modin.pandas as pd
# except ModuleNotFoundError:
#     import pandas as pd
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import numpy as np
import sklearn
import swifter # Not an unused import

from dataset.common import utils, mapping_utils, logger
from dataset.dse.dla_dataset_class import DlaDataset

def identity_fn(x):
    return x

def parallel_process_stuff(chunk, process_mappings):
    chunk.reset_index(inplace=True)
    if "arr" in process_mappings or "split" in process_mappings:
        # Process mappings into array representation
        # Does not reduce data size
        chunk["mapping.flat_mapping"] = chunk[["mapping.mapping", "prob.shape"]].swifter.apply(
            func=lambda x: mapping_utils.process_mapping(*x), axis=1)

        if "split" in process_mappings:
            # Use slower handler if multiple prob shapes
            if chunk["prob.shape"].nunique() > 1:
                # TODO: broken on new pandas version, needs fix
                chunk = chunk.swifter.apply(split_flat_mapping, axis=1)
            else: # faster version if all same prob shape
                num_mem_lvls = mapping_utils.get_num_mem_lvls(chunk.at[0, "mapping.mapping"])
                col_names = get_mapping_columns(chunk.at[0, "prob.shape"], num_mem_lvls)
                chunk[col_names] = pd.DataFrame(chunk["mapping.flat_mapping"].tolist())
    # Process prob shape
    chunk["prob.shape_cls"] = chunk["prob.shape"].swifter.apply(utils.prob_shape_to_class)

    return chunk

class DlaDatasetCreator():
    """Loads CSV to create DLA datasets.
    
    Extends torch.utils.data.Dataset
    """

    def __init__(self, 
                 df=None,
                 dataset_path=None, # TODO: make optional (currently required if normalization used)
                 stats_path=None,
                 stats=None,
                 split_ratios={"train":0.75, "valid":0.1, "test": 0.15},
                 total_samples=0,
                 shuffle=True,
                 target_log=True, target_norm="mean",
                 probfeat_log=True, probfeat_norm="mean",
                 archfeat_log=True, archfeat_norm="mean",
                 mapfeat_log=False, mapfeat_norm="mean",
                 process_mappings="",
                 process_mappings_obj="edp",
                 layer_count: pathlib.Path=None,
                 num_processes: int=1,):
        """
        Args:
            df (pd.DataFrame): Dataframe of data to use.
            dataset_path (string): Path to the csv file with data.
            stats_path (string | pathlib.Path): Path to previously constructed
                normalization stats file.
            split_ratios (dict[int]): Proportions (adding to 1 for training,
                validation, and test sets). Any value not provided will be
                inferred. Default is a 75-10-15 train-valid-test split.
                Example format: {"train": 0.75, "valid": 0.1, "test": 0.15}.
            total_samples (int): Total number of rows to use from CSV file,
                AFTER post-processing. 0 or any other false value indicates
                all rows should be used.
            <feature_type>_log: Boolean determining whether to take the log
                base 2 of a set of features (target, arch, prob, mapping)
            <feature_type>_norm: String type of normalization to apply to
                a set of features. Options: [mean, max] where mean applies
                mean-std normalization and max applies min-max normalization.
                Any false value results in no normalization.
            process_mappings: String determining how to process different 
                mappings for the same arch/prob. If you would like to, for
                example, get the min mapping and split it into Pandas columns,
                use the + character, e.g. "min+split". Options:
                    min: get min for a given objective (see process_mappings_obj)
                    arr: turns shorthand string into a flattened Python list
                    split: splits each factor into its own Pandas DF column
            process_mappings_obj: String determining to which performance
                metric to apply the above reduction [cycle, energy, edp, area]
        """
        self.params = locals()
        logger.debug(self.params)
        # Save config
        params_hash = copy.deepcopy(self.params)
        params_hash.pop("self")
        params_hash.pop("num_processes")
        params_hash = str(params_hash)
        # params_hash = str({
        #     "dataset_path": dataset_path,
        #     "split_ratios": split_ratios,
        #     "shuffle": shuffle,
        #     "total_samples": total_samples,
        #     "target_log": target_log,
        #     "target_norm": target_norm,
        #     "probfeat_log": probfeat_log,
        #     "probfeat_norm": probfeat_norm,
        #     "archfeat_log": archfeat_log,
        #     "archfeat_norm": archfeat_norm,
        #     "mapfeat_log": mapfeat_log,
        #     "mapfeat_norm": mapfeat_norm,
        #     "process_mappings": process_mappings,
        #     "process_mappings_obj": process_mappings_obj,
        # })

        # Save normalization stats for later
        # Or load existing stats
        skip_processing = False
        if dataset_path is None:
            # TODO: figure out whether/where to store stats if DF is passed in
            dataset_tmp_dir = pathlib.Path("dla_dataset")
            dataset_tmp_dir.mkdir(exist_ok=True)
            dataset_path = dataset_tmp_dir / "dataset.csv"
        dataset_path = pathlib.Path(dataset_path).resolve()
        dataset_name = dataset_path.stem # Gets filename without extension
        if not stats_path:
            self.stats_path = dataset_path.parent / f"dataset_stats_{dataset_name}.json"
        else:
            self.stats_path = pathlib.Path(stats_path).resolve()
        
        try:
            outer_stats = utils.parse_json(self.stats_path)
        except:
            outer_stats = {}
        if stats:
            pass
        elif params_hash in outer_stats:
            stats = outer_stats[params_hash]
            # if stats.get("train_parquet"):
            #     skip_processing = True
            #     logger.info("Same args, using saved data")
            #     parquet_start_time = time.time()
            #     train_df = pd.read_parquet(open(stats["train_parquet"], "rb"))
            #     valid_df = pd.read_parquet(open(stats["valid_parquet"], "rb"))
            #     test_df = pd.read_parquet(open(stats["test_parquet"], "rb"))
            #     logger.info("Loading .parquet files took %.2f seconds", time.time() - parquet_start_time)
        else:
            stats = {}

        if not skip_processing:
            if df is not None:
                full_dataset = df
            elif dataset_path is not None:
                dataset_path = pathlib.Path(dataset_path)
                logger.info("Loading dataset from CSV file: %s", dataset_path)
                load_csv_start_time = time.time()
                full_dataset = pd.read_csv(dataset_path)
                logger.info("Loading CSV took %.2f seconds", time.time() - load_csv_start_time)
            else:
                logger.error("Provide df or dataset_path to construct dataset")
                return

            if shuffle:
                full_dataset = sklearn.utils.shuffle(full_dataset, random_state=utils.get_random_seed())
                full_dataset.reset_index(drop=True, inplace=True)

            # Figure out what columns there are
            log_keys = []
            mean_norm_keys = []
            max_norm_keys = []
            min_max_norm_keys = []

            #######################################################################
            # Code below this comment modifies the DataFrame                      #
            #                                                                     #
            # Note that the order of operations below matters, since (e.g.)       #
            # normalization statistics are calculated after non-minimal mappings  #
            # are filtered out.                                                   #
            #######################################################################

            def reduce_rows(df, samps):
                df.reset_index(drop=True, inplace=True)
                if samps: # skip if total_samples=0 or any other false value
                    df = df[:total_samples]
                df.reset_index(drop=True, inplace=True)
                return df

            # Performance optimization: if we do not need the full data to process mappings, reduce
            # rows now
            rows_reduced = False
            if "min" not in process_mappings:
                full_dataset = reduce_rows(full_dataset, total_samples)
                rows_reduced = True

            orig_archfeat_keys = utils.keys_by_type(full_dataset, "arch")
            if full_dataset.iloc[0].get("arch.name") == "gemmini":
                pe_dim = np.ceil(full_dataset["arch.pe"] ** 0.5)
                # first get bits, then get KB
                sp_size = full_dataset["arch.mem2_depth"] * full_dataset["arch.mem2_width"] * full_dataset.get("arch.mem2_instances", 1)
                sp_size = np.ceil(sp_size / 8 / 1024)
                acc_size = full_dataset["arch.mem1_depth"] * full_dataset["arch.mem1_width"] * full_dataset.get("arch.mem1_instances", 1)
                acc_size = np.ceil(acc_size / 8 / 1024)
                full_dataset["arch.pe_dim"] = pe_dim
                full_dataset["arch.sp_size"] = sp_size
                full_dataset["arch.acc_size"] = acc_size
                full_dataset = full_dataset.drop(orig_archfeat_keys, axis=1)

            # Process mappings
            # May reduce data size
            if "min" in process_mappings:
                # Get min <metric> of each arch/prob combination
                by = utils.keys_by_type(full_dataset, "arch") + utils.keys_by_type(full_dataset, "prob")
                min_idxs = full_dataset.groupby(by=by)[f"target.{process_mappings_obj}"].idxmin()
                full_dataset = full_dataset.loc[min_idxs].reset_index()
            # Now we can cut down to desired number of samples
            if not rows_reduced:
                full_dataset = reduce_rows(full_dataset, total_samples)

            mappings_start_time = time.time()
            if "arr" in process_mappings or "split" in process_mappings:
                if num_processes > 1:
                    pool = multiprocessing.Pool(processes=num_processes)
                    chunks = np.array_split(full_dataset, num_processes)
                    funcs = []
                    for chunk in chunks:
                        f = pool.apply_async(parallel_process_stuff, (chunk, process_mappings,))
                        funcs.append(f)
                    full_dataset = pd.DataFrame([])
                    for f in funcs:
                        full_dataset = pd.concat([full_dataset, f.get()], ignore_index=True)
                else:
                    full_dataset = parallel_process_stuff(full_dataset, process_mappings)
            logger.info("Processing mappings took %.2f seconds", time.time() - mappings_start_time)

            full_dataset = full_dataset.copy()

            ########### Stop modifying self.df columns below this line ###########

            # Store keys for PPA targets to be log/normalized
            target_keys = utils.keys_by_type(full_dataset, "target")
            if target_log:
                log_keys.extend(target_keys)
            if target_norm == "mean":
                mean_norm_keys.extend(target_keys)
            elif target_norm == "max":
                max_norm_keys.extend(target_keys)
            elif target_norm == "minmax":
                min_max_norm_keys.extend(target_keys)

            # Store keys for hardware architectural features to be log/normalized
            archfeat_keys = utils.keys_by_type(full_dataset, "arch")
            if archfeat_log:
                log_keys.extend(archfeat_keys)
            if archfeat_norm == "mean":
                mean_norm_keys.extend(archfeat_keys)
            elif archfeat_norm == "max":
                max_norm_keys.extend(archfeat_keys)
            elif archfeat_norm == "minmax":
                min_max_norm_keys.extend(archfeat_keys)

            # # Explode prob instance dict string into separate columns
            # start_time = time.time()
            # self.df["instance"].map(eval)
            # logger.info("Mapping dict string to dict took %s seconds", time.time() - start_time)
            # instance_df = pd.DataFrame(self.df.pop("instance").tolist())
            # # instance_df = pd.json_normalize(self.df["instance"]) # Works for deep dicts
            # probfeat_keys = instance_df.keys() # TODO: actually get keys
            # self.df = self.df.join(instance_df)

            # Store keys for prob features to be log/normalized
            probfeat_keys = utils.keys_by_type(full_dataset, "prob")
            if probfeat_log:
                log_keys.extend(probfeat_keys)
            if probfeat_norm == "mean":
                mean_norm_keys.extend(probfeat_keys)
            elif probfeat_norm == "max":
                max_norm_keys.extend(probfeat_keys)
            elif probfeat_norm == "minmax":
                min_max_norm_keys.extend(probfeat_keys)

            # Store keys for mapping features to be log/normalized
            mapfeat_keys = utils.keys_by_type(full_dataset, "mapping")
            if mapfeat_log:
                log_keys.extend(mapfeat_keys)
            if mapfeat_norm == "mean":
                mean_norm_keys.extend(mapfeat_keys)
            elif mapfeat_norm == "max":
                max_norm_keys.extend(mapfeat_keys)
            elif mapfeat_norm == "minmax":
                min_max_norm_keys.extend(mapfeat_keys)

            # Actually apply log/normalization
            # Note log is applied before normalization
            log_start_time = time.time()
            for key in log_keys:
                try:
                    if any(full_dataset[key] < 0):
                        logger.warning("Key %s contains negative values, will be set to -1 during log", key)
                    if any(full_dataset[key] == 0):
                        logger.warning("Key %s contains zero values, will be set to -1 during log", key)
                    # Zeros stay as zero
                    full_dataset[key] = np.log2(full_dataset[key], out=np.zeros_like(full_dataset[key], dtype=np.float64)-1, where=full_dataset[key]>0)
                    stats[f"{key}_log"] = True
                except Exception as e:
                    logger.error("Could not take log of column '%s', exception %s", key, e)
            logger.info("Taking log took %.2f seconds", time.time() - log_start_time)

            # Train/validation/test split
            num_rows = len(full_dataset)
            train_ratio = split_ratios.get("train", 0.75)
            test_ratio = split_ratios.get("test", 0.15)
            test_ratio = min(1-train_ratio, test_ratio) # if train_ratio is greater than 0.85, just take rest of data
            valid_ratio = split_ratios.get("valid", 1 - train_ratio - test_ratio)
            
            train_part = int(train_ratio * num_rows)
            test_part = int(test_ratio * num_rows)
            # ensure the 3 parts do not overlap
            valid_part = min(num_rows - train_part - test_part, int(valid_ratio * num_rows))
            train_df = full_dataset[0: train_part].reset_index()
            valid_df = full_dataset[train_part:train_part+valid_part].reset_index()
            test_df = full_dataset[train_part+valid_part:train_part+valid_part+test_part].reset_index()
            # if split == "train":
            #     self.df = train_df
            # elif split == "valid":
            #     self.df = valid_df
            # elif split == "test":
            #     self.df = test_df
            logger.info("num_rows %d, train %d, valid %d, test %d",
                num_rows, len(train_df), len(valid_df), len(test_df))
            # Don't use full_dataset below this point

            for key in mean_norm_keys:
                try:
                    mean = stats.get(f"{key}_mean", train_df[key].mean())
                    std = stats.get(f"{key}_std", train_df[key].std().round(decimals=10))
                    stats[f"{key}_mean"] = mean
                    stats[f"{key}_std"] = std
                    if std == 0:
                        train_df[key] = train_df[key] - mean
                        valid_df[key] = valid_df[key] - mean
                        test_df[key] = test_df[key] - mean
                    else:
                        train_df[key] = (train_df[key] - mean) / std
                        valid_df[key] = (valid_df[key] - mean) / std
                        test_df[key] = (test_df[key] - mean) / std
                except Exception as e:
                    logger.error("Could not mean-std normalize column '%s', exception %s", key, e)
            norm_start_time = time.time()
            for key in min_max_norm_keys:
                try:
                    col_min = stats.get(f"{key}_min", train_df[key].min())
                    col_max = stats.get(f"{key}_max", train_df[key].max())
                    std = stats.get(f"{key}_std", train_df[key].std().round(decimals=10))
                    stats[f"{key}_min"] = float(col_min)
                    stats[f"{key}_max"] = float(col_max)
                    stats[f"{key}_std"] = std
                    if col_min == col_max:
                        if col_max != 0:
                            train_df[key] = train_df[key] / col_max
                            valid_df[key] = valid_df[key] / col_max
                            test_df[key] = test_df[key] / col_max
                    else:
                        train_df[key] = (train_df[key] - col_min) / (col_max - col_min)
                        valid_df[key] = (valid_df[key] - col_min) / (col_max - col_min)
                        test_df[key] = (test_df[key] - col_min) / (col_max - col_min)
                except Exception as e:
                    logger.error("Could not min-max normalize column '%s', exception %s", key, e)
            for key in max_norm_keys:
                try:
                    col_max = stats.get(f"{key}_max", train_df[key].max())
                    std = stats.get(f"{key}_std", train_df[key].std().round(decimals=10))
                    stats[f"{key}_max"] = float(col_max)
                    stats[f"{key}_std"] = std
                    if col_max != 0:
                        train_df[key] = train_df[key] / col_max
                        valid_df[key] = valid_df[key] / col_max
                        test_df[key] = test_df[key] / col_max
                except Exception as e:
                    logger.error("Could not min-max normalize column '%s', exception %s", key, e)
            logger.info("Normalizing took %.2f seconds", time.time() - norm_start_time)

            stats["params"] = params_hash
            train_path = dataset_path.parent / f"dataset_{dataset_name}_{utils.unique_filename('')}_train.parquet"
            valid_path = dataset_path.parent / f"dataset_{dataset_name}_{utils.unique_filename('')}_valid.parquet"
            test_path = dataset_path.parent / f"dataset_{dataset_name}_{utils.unique_filename('')}_test.parquet"
            stats["train_parquet"] = str(train_path)
            stats["valid_parquet"] = str(valid_path)
            stats["test_parquet"] = str(test_path)
            # train_df.to_parquet(open(train_path, "wb"), compression=None)
            # valid_df.to_parquet(open(valid_path, "wb"), compression=None)
            # test_df.to_parquet(open(test_path, "wb"), compression=None)

        self.train_data = DlaDataset(train_df, self.stats_path, self)
        self.valid_data = DlaDataset(valid_df, self.stats_path, self)
        self.test_data = DlaDataset(test_df, self.stats_path, self)

        outer_stats[params_hash] = stats
        utils.store_json(self.stats_path, outer_stats, indent=4)
        self.stats = stats
        self.outer_stats = outer_stats

    # wrap your csv importer in a function that can be mapped
    def read_csv(filename):
        'converts a filename to a pandas dataframe'
        return pd.read_csv(filename)

    def get_train_data(self):
        return self.train_data
    def get_valid_data(self):
        return self.valid_data
    def get_test_data(self):
        return self.test_data

def split_flat_mapping(s: pd.Series) -> pd.Series:
    # logger.error("%s", s)
    # logger.error("%s", s["mapping.mapping"])
    # exit(0)
    num_mem_lvls = mapping_utils.get_num_mem_lvls(s["mapping.mapping"])
    col_names = get_mapping_columns(s["prob.shape"], num_mem_lvls)
    flat_mapping = s["mapping.flat_mapping"]
    s = s.reindex(columns=s.columns.tolist() + col_names)
    s[col_names] = flat_mapping
    return s

def get_mapping_columns(prob_shape: str, num_mem_lvls: int) -> List[str]:
    """Get flattened list with column titles for the corresponding flattened mapping.
    """
    dims = utils.get_prob_dims(prob_shape)
    col_names = []
    # Count down mem lvls
    for mem_lvl in reversed(range(num_mem_lvls)):
        for factor_type in ["spatial", "temporal", "perm"]:
            for dim in dims:
                name = f"mapping.{factor_type}_L{mem_lvl}_{dim}"
                col_names.append(name)
    return col_names
