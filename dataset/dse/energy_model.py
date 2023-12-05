import multiprocessing
import traceback
import pathlib

import numpy as np
import pandas as pd
import torch

from dataset.common import logger, utils, mapping_utils
from dataset.dse import pytorch_util, eval, DlaDataset

class EnergyModel():
    def __init__(self, output_dir, arch_param_size):
        self.output_dir = output_dir
        self.arch_param_size = arch_param_size
        self.mlp = None
        self.loss_fn = torch.nn.MSELoss()

    def train(self, train_data: DlaDataset, valid_data: DlaDataset=None, num_iters=1000, gpu_id=0, continue_training=False, with_cache=False):
        """Train energy per access based on arch, multiplied by access counts
        """
        self.train_data = train_data
        self.stats = train_data.creator.stats
        self.with_cache = with_cache

        access_key_types = ["dse.access"]
        if "dse.access_mac" not in train_data.df.columns:
            # populate access counts
            pytorch_util.init_gpu(False)
            add_accesses_col(self.output_dir, train_data, self.with_cache)
            access_keys = []
            for key_type in access_key_types:
                type_keys = utils.keys_by_type(train_data.df, key_type, scalar_only=True)
                access_keys.extend(type_keys)
            pytorch_util.init_gpu(gpu_id=gpu_id)
            for access_key in access_keys:
                col_max = train_data.df[access_key].max()
                mean = train_data.df[access_key].mean()
                std = train_data.df[access_key].std()
                # if std != 0:
                #     train_data.df[access_key] = (train_data.df[access_key] - mean) / std
                # else:
                #     train_data.df[access_key] = (train_data.df[access_key] - mean)
                if col_max != 0:
                    train_data.df[access_key] = (train_data.df[access_key] / col_max)
                self.stats[access_key + "_max"] = col_max
                # self.stats[access_key + "_mean"] = mean
                # self.stats[access_key + "_std"] = std
            utils.store_json(train_data.creator.stats_path, train_data.creator.outer_stats, indent=4)
            # train_data.df.to_parquet(open(train_data.creator.stats["train_parquet"], "wb"), compression=None)
            print(train_data.df["dse.access_mac"])
        else:
            access_keys = []
            for key_type in access_key_types:
                type_keys = utils.keys_by_type(train_data.df, key_type, scalar_only=True)
                access_keys.extend(type_keys)

        # # add spatial tiling product
        # train_data.df["dse.spatial_prod"] = train_data.df["mapping.spatial_L1_C"] * train_data.df["mapping.spatial_L2_K"]
        # if valid_data is not None:
        #     valid_data.df["dse.spatial_prod"] = valid_data.df["mapping.spatial_L1_C"] * valid_data.df["mapping.spatial_L2_K"]

        if valid_data and ("dse.access_mac" not in valid_data.df.columns):
            # populate access counts
            pytorch_util.init_gpu(False)
            add_accesses_col(self.output_dir, valid_data, self.with_cache)
            pytorch_util.init_gpu(gpu_id=gpu_id)
            for access_key in access_keys:
                valid_data.creator.stats[access_key + "_max"] = self.stats[access_key + "_max"]
                col_max = self.stats[access_key + "_max"]
                if col_max != 0:
                    valid_data.df[access_key] = (valid_data.df[access_key] / col_max)
            # valid_data.df.to_parquet(open(self.stats["test_parquet"], "wb"), compression=None)
            print(valid_data.df["dse.access_mac"])

        x_key_types = ["arch"]# + relevant_keys
        x_keys = []
        for key_type in x_key_types:
            type_keys = utils.keys_by_type(train_data.df, key_type, scalar_only=True)
            x_keys.extend(type_keys)

        y_keys = ["target.energy"]

        # X_train = train_data.df[x_keys].to_numpy()
        # X_train = train_data.denorm("arch", X_train).numpy()
        # access_train = train_data.df[access_keys].to_numpy()
        y_train = train_data.df[y_keys].to_numpy()
        y_train = train_data.denorm("target.energy", y_train).numpy()
        self.energy_max = y_train.max() / 100
        # y_train = y_train / self.energy_max
        # X_train = pytorch_util.from_numpy(X_train)
        # access_train = pytorch_util.from_numpy(access_train)
        # y_train = pytorch_util.from_numpy(y_train)
        # # X_train = torch.cat((X_train, torch.square(X_train)), dim=1)
        # self.arch_max = X_train.max(dim=0)[0] / 100
        # X_train = X_train / self.arch_max

        logger.debug("energy_max: %s", self.energy_max)
        # logger.debug("arch_max: %s", self.arch_max)

        mac_energy = 0.5608e-6 / self.energy_max * self.stats["dse.access_mac_max"]
        reg_energy = 0.48746172e-6 / self.energy_max * self.stats["dse.access_memlvl0_max"]
        dram_energy = 100e-6 / self.energy_max * self.stats["dse.access_memlvl3_max"]
        self.mac_energy = pytorch_util.from_numpy(np.array([mac_energy]))
        self.reg_energy = pytorch_util.from_numpy(np.array([reg_energy]))
        self.dram_energy = pytorch_util.from_numpy(np.array([dram_energy]))

        return

        save_str = "energy_arch_only_denormed"
        mlp_paths = []
        opt_path = pathlib.Path(self.output_dir).resolve() / f"mlp_opt_{save_str}.pt"

        mlps = []
        arch_idxs = [None, None, [0, 2], [1], None]
        # arch_idxs = [None, None, [0, 2, 3, 5], [1, 4], None]
        # arch_idxs = [[0, 1, 2, 3, 4, 5]]
        self.arch_idxs = []
        for i, access_key in enumerate(access_keys):
            if access_key == "dse.access_mac" or access_key == "dse.access_memlvl0" or access_key == "dse.access_memlvl3":
                continue
        # for i in range(1):
            # hidden_layer_sizes = (32,128,32,) # in 2_9_23 output_dir
            hidden_layer_sizes = (8,32,) # in debug output_dir
            mlp = pytorch_util.build_mlp(
                input_size=len(arch_idxs[i]),
                output_size=1,
                n_layers=len(hidden_layer_sizes),
                size=hidden_layer_sizes,
                activation="relu",
                dropout=0.4,
                output_activation="softplus",
            )
            mlp.to(pytorch_util.device)
            mlps.append(mlp)
            mlp_path = pathlib.Path(self.output_dir).resolve() / f"mlp_{save_str}_{i}.pt"
            mlp_paths.append(mlp_path)
            self.arch_idxs.append(arch_idxs[i])

        params = []
        for mlp in mlps:
            params += list(mlp.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.2)

        self.mlps = mlps
        self.optimizer = optimizer

        try:
            for i, mlp in enumerate(mlps):
                mlp.load_state_dict(torch.load(mlp_paths[i]))
            # optimizer.load_state_dict(torch.load(opt_path))
            logger.info("Loaded existing MLP and optimizer from %s and %s", mlp_path, opt_path)
            if not continue_training:
                return
            self.unfreeze()
        except:
            print(traceback.format_exc())
            pass

        X_train = X_train / self.arch_max
        train_dataset = pytorch_util.X_y_dataset(X_train, y_train, access_train)
        batch_size = 40000
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        mapping_keys = utils.keys_by_type(train_data.df, "mapping")
        relevant_keys = []
        for idx, key in enumerate(mapping_keys):
            if stats[key+"_std"] != 0:
                relevant_keys.append(key)

        logger.info("Training prediction of target(s) %s with inputs %s for %s iterations, %s data points",
                    y_keys, x_key_types, num_iters, len(X_train))
        mlp.train()
        for iter in range(num_iters):
            for X_batch, y_batch, access_batch in train_data_loader:
                coeff_batch = self.predict_coeff(X_batch)
                y_pred_batch = (coeff_batch * access_batch).sum(dim=1).unsqueeze(-1)
                loss = self.loss_fn(y_pred_batch, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.1)
                optimizer.step()
            logger.info(f"Finished training iter {iter}, loss {loss}")
            logger.debug("y_batch: %s, y_pred_batch: %s, X_batch: %s, coeff_batch: %s", 
                         y_batch[-1], y_pred_batch[-1], X_batch[-1], coeff_batch[-1])
            scheduler.step()
            if (iter+1) % 20 == 0:
                for i, mlp in enumerate(mlps):
                    torch.save(mlp.state_dict(), mlp_paths[i])
                torch.save(optimizer.state_dict(), opt_path)
        for i, mlp in enumerate(mlps):
            torch.save(mlp.state_dict(), mlp_paths[i])
        torch.save(optimizer.state_dict(), opt_path)

    def predict(self, arch_params, access_params):
        # dram_access_params = access_params[:,-1].unsqueeze(-1)
        # dram_width = 64 / self.stats["dse.access_memlvl3_max"]
        # dram_access_params = torch.ceil(dram_access_params / dram_width) * dram_width
        # access_params = torch.cat((access_params[:,:-1], dram_access_params), dim=1)
        pred = self.predict_coeff(arch_params)
        return (pred * access_params).sum(dim=1).unsqueeze(-1)

    def predict_coeff(self, arch_params):
        """arch_params should be denormalized
        """
        squeeze_at_end = False
        if len(arch_params.size()) == 1:
            squeeze_at_end = True
            arch_params = arch_params.unsqueeze(0)
        # pred = None
        # for i, mlp in enumerate(self.mlps):
        #     mlp_pred = mlp(arch_params[:,self.arch_idxs[i]])
        #     if pred is None:
        #         pred = mlp_pred
        #     else:
        #         pred = torch.cat((pred, mlp_pred), dim=1)
        mac_col = self.mac_energy.expand((len(arch_params), 1))
        reg_col = self.reg_energy.expand((len(arch_params), 1))
        acc_col = 1e-6 * (0.1005 * arch_params[:,2] / arch_params[:,0] + 1.94) / self.energy_max * self.stats["dse.access_memlvl1_max"]
        acc_col = acc_col.unsqueeze(-1)
        # acc_r_col = 1e-6 * (0.1005 * arch_params[:,2] / arch_params[:,0] + 1.94) / self.energy_max * self.stats["dse.access_memlvl1_r_max"]
        # acc_r_col = acc_r_col.unsqueeze(-1)
        # acc_w_col = 1e-6 * (0.1005 * arch_params[:,2] / arch_params[:,0] + 1.94) / self.energy_max * self.stats["dse.access_memlvl1_w_max"]
        # acc_w_col = acc_w_col.unsqueeze(-1)
        sp_col = 1e-6 * (0.02513 * arch_params[:,1] + 0.49) / self.energy_max * self.stats["dse.access_memlvl2_max"]
        sp_col = sp_col.unsqueeze(-1)
        # sp_r_col = 1e-6 * (0.02513 * arch_params[:,1] + 0.49) / self.energy_max * self.stats["dse.access_memlvl2_r_max"]
        # sp_r_col = sp_r_col.unsqueeze(-1)
        # sp_w_col = 1e-6 * (0.02513 * arch_params[:,1] + 0.49) / self.energy_max * self.stats["dse.access_memlvl2_w_max"]
        # sp_w_col = sp_w_col.unsqueeze(-1)
        dram_col = self.dram_energy.expand((len(arch_params), 1))
        pred = torch.cat((mac_col, reg_col, acc_col, sp_col, dram_col), dim=1)
        # pred = torch.cat((mac_col, reg_col, acc_r_col, acc_w_col, sp_r_col, sp_w_col, dram_col), dim=1)
        if squeeze_at_end:
            pred = pred.squeeze(0)
        return pred

    def norm_energy(self, energy):
        return energy / self.energy_max
    def denorm_energy(self, energy):
        return energy * self.energy_max

    def freeze(self):
        for mlp in self.mlps:
            mlp.eval()
            for p in mlp.parameters():
                p.requires_grad = False
    
    def unfreeze(self):
        for mlp in self.mlps:
            mlp.train()
            for p in mlp.parameters():
                p.requires_grad = True

    def test(self, test_data: DlaDataset, num_worst_points=10, gpu_id=0):
        x_key_types = ["arch"]
        x_keys = []
        for key_type in x_key_types:
            type_keys = utils.keys_by_type(test_data.df, key_type, scalar_only=True)
            x_keys.extend(type_keys)

        stats = test_data.creator.stats
        access_key_types = ["dse.access"]
        if "dse.access_mac" not in test_data.df.columns:
            # populate access counts
            pytorch_util.init_gpu(False)
            add_accesses_col(self.output_dir, test_data, self.with_cache)
            access_keys = []
            for key_type in access_key_types:
                type_keys = utils.keys_by_type(test_data.df, key_type, scalar_only=True)
                access_keys.extend(type_keys)
            pytorch_util.init_gpu(gpu_id=gpu_id)
            for access_key in access_keys:
                # if std != 0:
                #     test_data.df[access_key] = (test_data.df[access_key] - mean) / std
                # else:
                #     test_data.df[access_key] = (test_data.df[access_key] - mean)
                if mean != 0:
                    test_data.df[access_key] = (test_data.df[access_key] / stats[access_key + "_max"])
            utils.store_json(test_data.creator.stats_path, test_data.creator.outer_stats, indent=4)
            # test_data.df.to_parquet(open(test_data.creator.stats["train_parquet"], "wb"), compression=None)
            print(test_data.df["dse.access_mac"])
        else:
            access_keys = []
            for key_type in access_key_types:
                type_keys = utils.keys_by_type(test_data.df, key_type, scalar_only=True)
                access_keys.extend(type_keys)

        y_keys = ["target.energy"]

        arch_test = test_data.df[x_keys]
        y_test = test_data.df[y_keys]
        access_test = test_data.df[access_keys]
        access_test_denormed = access_test.copy()
        for access_key in access_keys:
            mean = stats[access_key + "_max"]
            logger.debug("%s: %s", access_key + "_max", mean)
            if mean != 0:
                access_test_denormed[access_key] = access_test_denormed[access_key] * mean

        arch_test = pytorch_util.from_numpy(arch_test.to_numpy())
        arch_test_denormed = test_data.denorm("arch", arch_test)
        y_test = pytorch_util.from_numpy(y_test.to_numpy())
        y_test = test_data.denorm("target.energy", y_test) / self.energy_max
        access_test = pytorch_util.from_numpy(access_test.to_numpy())
        access_test_denormed = torch.tensor(access_test_denormed.to_numpy())
        test_dataset = pytorch_util.X_y_dataset(arch_test, y_test, access_test, arch_test_denormed)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40000)
        y_pred = None
        coeff_pred = None
        for arch_batch, _, access_batch, arch_batch_denormed in test_data_loader:
            y_batch_pred = self.predict(arch_batch_denormed, access_batch)
            coeff_batch_pred = self.predict_coeff(arch_batch_denormed)
            if y_pred is None:
                y_pred = y_batch_pred
                coeff_pred = coeff_batch_pred
            else:
                y_pred = torch.cat((y_pred, y_batch_pred), dim=0)
                coeff_pred = torch.cat((coeff_pred, coeff_batch_pred), dim=0)
        
        logger.info("min coeff: %s", coeff_pred.min())
        logger.info("max coeff: %s", coeff_pred.max())
        losses = []
        for col in range(y_test.size(1)):
            col_loss = self.loss_fn(y_test[:,col], y_pred[:,col])
            logger.info("Col %s loss: %s", y_keys[col], col_loss)
            losses.append(col_loss)

            loss_fn_no_red = torch.nn.MSELoss(reduction="none")
            if num_worst_points > 0:
                col_loss_no_red = loss_fn_no_red(y_test[:,col], y_pred[:,col])
                col_worst_losses, col_worst_idxs = col_loss_no_red.topk(num_worst_points)
                logger.info("Col %s worst points idxs: %s", y_keys[col], col_worst_idxs)
                logger.info("Col %s worst points losses: %s", y_keys[col], col_worst_losses)
                logger.info("Col %s worst points X-vals: %s", y_keys[col], arch_test[col_worst_idxs])
                logger.info("Col %s worst points X-vals (denormed): %s", y_keys[col], test_data.denorm(x_key_types, arch_test[col_worst_idxs]))
                logger.info("Col %s worst points coeff-preds: %s", y_keys[col], coeff_pred[col_worst_idxs])
                logger.info("Col %s worst points access-vals: %s", y_keys[col], access_test[col_worst_idxs])
                logger.info("Col %s worst points access-vals (denormed): %s", y_keys[col], access_test_denormed[col_worst_idxs])
                logger.info("Col %s worst points Y-vals: %s", y_keys[col], y_test[:,col][col_worst_idxs])
                logger.info("Col %s worst points Y-preds: %s", y_keys[col], y_pred[:,col][col_worst_idxs])
                logger.info("Col %s worst points data: %s", y_keys[col], test_data.df.reset_index(drop=True).iloc[pytorch_util.to_numpy(col_worst_idxs)].to_string())

        y_test = pytorch_util.to_numpy(y_test)
        y_pred = pytorch_util.to_numpy(y_pred)
        return y_test, y_pred, losses

def add_col(s: pd.Series, output_dir, prob_keys, dataset, with_cache):
    mapping = s["mapping.flat_mapping"]
    prob_feats = list(s[prob_keys].values)
    prob_feats = dataset.denorm("prob", prob_feats)
    prob = eval.parse_prob(output_dir, prob_keys, prob_feats)
    reads, updates, writes = mapping_utils.accesses_from_mapping(mapping, prob, with_cache=with_cache)
    s[f"dse.access_mac"] = float(reads[0][0])
    relevant = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
    if with_cache:
        relevant.append([1,1,1])
    for mem_lvl in range(len(reads)):
        s[f"dse.access_memlvl{mem_lvl}"] = 0
        for tensor in range(len(reads[mem_lvl])):
            if relevant[mem_lvl][tensor]:
                s[f"dse.access_memlvl{mem_lvl}"] += float(reads[mem_lvl][tensor] + updates[mem_lvl][tensor] + writes[mem_lvl][tensor])
    # for mem_lvl in range(len(reads)):
    #     if mem_lvl == 1 or mem_lvl == 2:
    #         s[f"dse.access_memlvl{mem_lvl}_r"] = 0
    #         s[f"dse.access_memlvl{mem_lvl}_w"] = 0
    #     else:
    #         s[f"dse.access_memlvl{mem_lvl}"] = 0
    #     for tensor in range(len(reads[mem_lvl])):
    #         if relevant[mem_lvl][tensor]:
    #             # s[f"dse.access_reads_memlvl{mem_lvl}_tensor{tensor}"] = float(reads[mem_lvl][tensor])
    #             # s[f"dse.access_updates_memlvl{mem_lvl}_tensor{tensor}"] = float(updates[mem_lvl][tensor])
    #             # s[f"dse.access_writes_memlvl{mem_lvl}_tensor{tensor}"] = float(writes[mem_lvl][tensor])
    #             if mem_lvl == 1 or mem_lvl == 2:
    #                 s[f"dse.access_memlvl{mem_lvl}_r"] += float(reads[mem_lvl][tensor])
    #                 s[f"dse.access_memlvl{mem_lvl}_w"] += float(updates[mem_lvl][tensor] + writes[mem_lvl][tensor])
    #             else:
    #                 s[f"dse.access_memlvl{mem_lvl}"] += float(reads[mem_lvl][tensor] + updates[mem_lvl][tensor] + writes[mem_lvl][tensor])
    return s
def parallel_apply(chunk, output_dir, prob_keys, dataset, with_cache):
    return chunk.swifter.apply(add_col, args=(output_dir, prob_keys, dataset, with_cache), axis=1)
def add_accesses_col(output_dir, dataset, with_cache):
    """
    make sure it works with all normalization
    """
    df: pd.DataFrame = dataset.df
    prob_keys = utils.keys_by_type(df, "prob")
    # full_dataset = df.apply(add_col, axis=1, args=(output_dir, prob_keys, dataset))
    num_processes = 16
    pool = multiprocessing.Pool(processes=num_processes)
    chunks = np.array_split(dataset.df, num_processes)
    funcs = []
    for chunk in chunks:
        f = pool.apply_async(parallel_apply, (chunk,output_dir,prob_keys,dataset,with_cache))
        funcs.append(f)
    full_dataset = pd.DataFrame([])
    for i, f in enumerate(funcs):
        full_dataset = pd.concat([full_dataset, f.get()], ignore_index=True)
        funcs[i] = None

    # # debug mode #####
    # full_dataset = pd.DataFrame([])
    # for chunk in chunks:
    #     full_dataset = pd.concat([full_dataset, chunk.apply(add_col, axis=1, args=(output_dir,prob_keys,dataset,with_cache))], ignore_index=True)
    # ##################

    dataset.df = full_dataset
