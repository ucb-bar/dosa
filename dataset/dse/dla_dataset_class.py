import math

import torch
import numpy as np

from dataset.common import utils, logger

class DlaDataset(torch.utils.data.Dataset):
    def __init__(self, df, stats_path, creator):
        self.df = df
        self.stats_path = stats_path
        self.creator = creator
        self.setdtype = torch.float32

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        targets = self.df.loc[idx, utils.keys_by_type(self.df, "target")].astype(np.float64)
        arch_feats = self.df.loc[idx, utils.keys_by_type(self.df, "arch")].astype(np.float64)
        prob_feats = self.df.loc[idx, utils.keys_by_type(self.df, "prob")].astype(np.float64)
        map_feats = self.df.loc[idx, utils.keys_by_type(self.df, "mapping")].astype(np.float64)

        targets = np.array([targets]).reshape(-1, )
        targets = torch.tensor(targets, dtype=self.setdtype)
        arch_feats = np.array([arch_feats]).reshape(-1, )
        arch_feats = torch.tensor(arch_feats, dtype=self.setdtype)
        prob_feats = np.array([prob_feats]).reshape(-1, )
        prob_feats = torch.tensor(prob_feats, dtype=self.setdtype)
        map_feats = np.array([map_feats]).reshape(-1, )
        map_feats = torch.tensor(map_feats, dtype=self.setdtype)

        return targets, arch_feats, prob_feats, map_feats

    def norm(self, feat_type, feats):
        if isinstance(feat_type, list) or isinstance(feat_type, tuple):
            keys = []
            for type in feat_type:
                keys.extend(utils.keys_by_type(self.df, type))
        else:
            keys = utils.keys_by_type(self.df, feat_type)
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(np.array(feats), dtype=self.setdtype)
        squeeze_at_end = False
        if len(feats.size()) == 0:
            squeeze_at_end = True
            feats = feats.unsqueeze(0)
        if len(feats.size()) == 1:
            squeeze_at_end = True
            feats = feats.unsqueeze(0)
        stats = self.creator.stats
        adder_tensor = torch.zeros_like(feats).detach()
        multiplier_tensor = torch.ones_like(feats).detach()
        log = False

        for idx, key in enumerate(keys):
            if stats.get(f"{key}_log"):
                # # full_dataset[key] = np.log2(full_dataset[key], out=np.zeros_like(full_dataset[key], dtype=np.float64)-1, where=full_dataset[key]>0)
                # if feats[:,idx] <= 0:
                #     feats[:,idx] = -1
                # else:
                #     feats[:,idx] = torch.log2(feats[:,idx])
                log = True
            mean = stats.get(f"{key}_mean")
            std = stats.get(f"{key}_std")
            col_min = stats.get(f"{key}_min")
            col_max = stats.get(f"{key}_max")
            if mean is not None:
                adder_tensor[:,idx] = - mean
                if std != 0:
                    multiplier_tensor[:,idx] = 1 / std
            elif col_min is not None:
                if col_min == col_max:
                    if col_max != 0:
                        multiplier_tensor[:,idx] = 1 / col_max
                else:
                    adder_tensor[:,idx] = - col_min
                    multiplier_tensor[:,idx] = 1 / (col_max - col_min)
            elif col_max is not None:
                if col_max != 0:
                    multiplier_tensor[:,idx] = 1 / col_max

        output_feats = feats
        if log:
            output_feats = torch.log2(output_feats)
        output_feats = (output_feats + adder_tensor) * multiplier_tensor
        if squeeze_at_end:
            output_feats = output_feats.squeeze()
        return output_feats

    def denorm(self, feat_type, feats):
        if isinstance(feat_type, list) or isinstance(feat_type, tuple):
            keys = []
            for type in feat_type:
                keys.extend(utils.keys_by_type(self.df, type))
        else:
            keys = utils.keys_by_type(self.df, feat_type)
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(np.array(feats), dtype=self.setdtype)
        squeeze_at_end = False
        if len(feats.size()) == 0:
            squeeze_at_end = True
            feats = feats.unsqueeze(0)
        if len(feats.size()) == 1:
            squeeze_at_end = True
            feats = feats.unsqueeze(0)
        stats = self.creator.stats
        adder_tensor = torch.zeros_like(feats).detach()
        multiplier_tensor = torch.ones_like(feats).detach()

        exp = False
        for idx, key in enumerate(keys):
            mean = stats.get(f"{key}_mean")
            std = stats.get(f"{key}_std")
            col_min = stats.get(f"{key}_min")
            col_max = stats.get(f"{key}_max")
            if mean is not None:
                adder_tensor[:,idx] = mean
                if std != 0:
                    multiplier_tensor[:,idx] = std
            elif col_min is not None:
                if col_min == col_max:
                    multiplier_tensor[:,idx] = col_max
                else:
                    multiplier_tensor[:,idx] = (col_max - col_min)
                    adder_tensor[:,idx] = col_min
            elif col_max is not None:
                if col_max != 0:
                    multiplier_tensor[:,idx] = col_max

            if stats.get(f"{key}_log"):
                # if feats[:,idx] == -1:
                #     feats[:,idx] = 0
                # else:
                #     feats[:,idx] = torch.exp2(feats[:,idx])
                exp = True

        output_feats = feats
        output_feats = (output_feats * multiplier_tensor) + adder_tensor
        if exp:
            output_feats = torch.exp2(output_feats)
        if squeeze_at_end:
            output_feats = output_feats.squeeze()
        return output_feats
