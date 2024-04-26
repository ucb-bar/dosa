import multiprocessing
import traceback
import pathlib
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from dataset import DATASET_ROOT_PATH
from dataset.common import logger, utils, mapping_utils
from dataset.dse import pytorch_util, eval, predictors, DlaDataset

class LatencyModel():
    def __init__(self, output_dir, relevant_mapping_keys):
        self.output_dir = output_dir
        self.relevant_mapping_keys = relevant_mapping_keys
        self.mlps = None
        self.loss_fn = torch.nn.L1Loss()

    def train(self, train_data: DlaDataset, valid_data: DlaDataset=None, train_model=False, with_analytical=True, with_roofline=True, continue_training=False, num_iters=1000, gpu_id=0, interp_points=0, with_cache=False):
        """Train latency model.

        Assumes normalized access count values have already been appended to train_data.df by EnergyModel
        """
        self.train_data = train_data
        self.target_key = "target.cycle"
        self.train_model = train_model
        self.with_analytical = with_analytical
        if not with_analytical:
            with_roofline = False # roofline can only be used with analytical
        self.with_roofline = with_roofline
        self.with_cache = with_cache
        access_keys = utils.keys_by_type(train_data.df, "dse.access")
        # mapping_net_layers = (128, 256, 32)
        mapping_net_layers = (256, 1024, 256)
        # self.mlp = predictors.train_mlp(train_data, self.relevant_mapping_keys + access_keys + ["prob"], ["target.cycle"], hidden_layer_sizes=mapping_net_layers,
        #                                      output_dir=self.output_dir, save_str="conv_mapping", num_iters=num_iters, gpu_id=gpu_id, continue_training=continue_training,
        #                                      interp_points=len(train_data)//2,)
        # self.mlp = predictors.train_mlp(train_data, self.relevant_mapping_keys + ["prob"], ["target.cycle"], hidden_layer_sizes=mapping_net_layers,
        #                                      output_dir=self.output_dir, save_str="conv_mapping", num_iters=num_iters, gpu_id=gpu_id, continue_training=continue_training,
        #                                      interp_points=len(train_data)//2,)
        # self.mlp = predictors.train_mlp(train_data, access_keys, ["target.cycle"], hidden_layer_sizes=mapping_net_layers,
        #                                      output_dir=self.output_dir, save_str="conv_mapping", num_iters=num_iters, gpu_id=gpu_id, continue_training=continue_training,)

        if not train_model:
            return

        mapping_keys = self.relevant_mapping_keys

        self.internal_relevant_idxs = []
        self.internal_relevant_keys = []
        for idx, key in enumerate(mapping_keys):
            # if "mapping.temporal_L0_Q" in key:
            #     continue
            if train_data.creator.stats[key+"_std"] != 0:
                self.internal_relevant_idxs.append(idx)
                self.internal_relevant_keys.append(key)

        save_str = "latency"
        mlp_paths = []
        opt_path = pathlib.Path(self.output_dir).resolve() / f"mlp_opt_{save_str}.pt"

        mlps = []
        hidden_layer_sizes = (256, 512, 2048, 4096, 4096, 2048, 512, 256)        # hidden_layer_sizes = (256, 1024, 256)
        # hidden_layer_sizes = (16)
        # hidden_layer_sizes = (512, 2048, 2048, 512, 128)
        input_size = len(self.internal_relevant_idxs)+11+3
        if self.with_roofline:
            input_size = input_size + 5
        mlp = pytorch_util.build_mlp(
            input_size=input_size,
            output_size=1,
            n_layers=len(hidden_layer_sizes),
            size=hidden_layer_sizes,
            activation="gelu",
            dropout=0.3,
            output_activation="softplus",
        )
        mlp.to(pytorch_util.device)
        mlps.append(mlp)
        if self.with_analytical:
            pred_type = "both"
        else:
            pred_type = "dnn"
        mlp_paths.append(DATASET_ROOT_PATH / "dse" / "trained_models" / "artifact" / pred_type / f"mlp_{save_str}_0.pt")
        # mlp_paths.append(self.output_dir / f"mlp_{pred_type}_{save_str}_0.pt")

        # hidden_layer_sizes = (256, 64)
        # mlp = pytorch_util.build_mlp(
        #     input_size=len(self.relevant_mapping_keys)+11+1, # 11 prob feats
        #     output_size=1,
        #     n_layers=len(hidden_layer_sizes),
        #     size=hidden_layer_sizes,
        #     activation="relu",
        #     dropout=0.3,
        # )
        # mlp.to(pytorch_util.device)
        # mlps.append(mlp)
        # mlp_paths.append(pathlib.Path(self.output_dir).resolve() / f"mlp_{save_str}_1.pt")

        self.mlps = mlps

        params = []
        for mlp in mlps:
            params += list(mlp.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=1e-5)
        self.optimizer = optimizer
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.8)
    
        prob_keys = utils.keys_by_type(train_data.df, "prob", scalar_only=True)
        y_keys = [self.target_key]
        arch_keys = utils.keys_by_type(train_data.df, "arch")

        arch_train = train_data.df[arch_keys].to_numpy()
        mapping_train = train_data.df[mapping_keys].to_numpy()
        prob_train = train_data.df[prob_keys].to_numpy()
        access_train = train_data.df[access_keys].to_numpy()
        y_train = train_data.df[y_keys].to_numpy()

        # # remove worst points
        # num_worst_points = len(y_train) // 2
        # logger.info("Removing %s of %s worst training points", num_worst_points*3//4, num_worst_points)
        # mask = np.argpartition(np.squeeze(y_train), kth=-num_worst_points)[:-num_worst_points*3//4]
        # mapping_train = mapping_train[mask]
        # prob_train = prob_train[mask]
        # access_train = access_train[mask]
        # y_train = y_train[mask]

        # if interp_points > 0:
        #     logger.info("Adding %s interpolation points", interp_points)
        #     idx_pairs = np.random.randint(0, len(mapping_train), size=(2, interp_points*20))
        #     first_points = mapping_train[idx_pairs[0]]
        #     second_points = mapping_train[idx_pairs[1]]
        #     norms = np.linalg.norm(first_points - second_points, axis=1)
        #     mask = np.argpartition(norms, kth=interp_points)[:interp_points]
        #     ratios = np.expand_dims(np.random.rand(interp_points), -1)
        #     mapping_train_interp = ratios * mapping_train[idx_pairs[0]][mask] + (1-ratios) * mapping_train[idx_pairs[1]][mask]
        #     mapping_train = np.vstack((mapping_train, mapping_train_interp))
        #     prob_train_interp = ratios * prob_train[idx_pairs[0]][mask] + (1-ratios) * prob_train[idx_pairs[1]][mask]
        #     prob_train = np.vstack((prob_train, prob_train_interp))
        #     access_train_interp = ratios * access_train[idx_pairs[0]][mask] + (1-ratios) * access_train[idx_pairs[1]][mask]
        #     access_train = np.vstack((access_train, access_train_interp))
        #     y_train_interp = ratios * y_train[idx_pairs[0]][mask] + (1-ratios) * y_train[idx_pairs[1]][mask]
        #     y_train = np.vstack((y_train, y_train_interp))

        arch_train = pytorch_util.from_numpy(arch_train)
        mapping_train = pytorch_util.from_numpy(mapping_train)
        prob_train = pytorch_util.from_numpy(prob_train)
        access_train = pytorch_util.from_numpy(access_train)
        y_train = pytorch_util.from_numpy(y_train)

        if valid_data is not None:
            arch_valid = valid_data.df[arch_keys].to_numpy()
            mapping_valid = valid_data.df[mapping_keys].to_numpy()
            prob_valid = valid_data.df[prob_keys].to_numpy()
            access_valid = valid_data.df[access_keys].to_numpy()
            y_valid = valid_data.df[y_keys].to_numpy()

            arch_valid = pytorch_util.from_numpy(arch_valid)
            mapping_valid = pytorch_util.from_numpy(mapping_valid)
            prob_valid = pytorch_util.from_numpy(prob_valid)
            access_valid = pytorch_util.from_numpy(access_valid)
            y_valid = pytorch_util.from_numpy(y_valid)

        rooflines = self.gen_rooflines(arch_train, mapping_train, access_train)
        self.roofline_max = rooflines.max()

        # try:
        #     for i, mlp in enumerate(mlps):
        #         mlp.load_state_dict(torch.load(mlp_paths[i]))
        #         logger.info("Loaded existing MLP from %s", mlp_paths[i])
        #     # optimizer.load_state_dict(torch.load(opt_path))
        #     if not continue_training:
        #         return
        #     self.unfreeze()
        # except:
        #     print(traceback.format_exc())
        #     pass

        # for mlp_path in mlp_paths:
        #     backup_mlp_path = mlp_path.parent / (mlp_path.name + ".bak")
        #     if mlp_path.exists():
        #         logger.info("Found existing MLP at %s, copying to %s", mlp_path, backup_mlp_path)
        #         shutil.copy(mlp_path, backup_mlp_path)

        train_dataset = pytorch_util.X_y_dataset(arch_train, mapping_train, prob_train, y_train, access_train)
        batch_size = 100000
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        logger.info("Training prediction of target(s) %s with inputs %s for %s iterations, %s data points",
                    y_keys, ["mapping", "prob", "access"], num_iters, len(train_dataset))
        # for mlp in mlps:
        #     mlp.train()
        for iter in range(num_iters):
            for arch_batch, mapping_batch, prob_batch, y_batch, access_batch in train_data_loader:
                # y_pred_batch = self.train_predict_noroofline(arch_batch, mapping_batch, access_batch, prob_batch)
                real_loss = 0
                if self.with_analytical:
                    y_pred_model = self.train_predict(arch_batch, mapping_batch, access_batch, prob_batch)
                    y_pred_analytical = self.predict_analytical(arch_batch, mapping_batch, access_batch)
                    # real_loss += torch.sum(torch.square(torch.clamp(y_pred_model - y_pred_analytical*100, min=0))) * 0.1
                    # real_loss += torch.sum(torch.clamp(y_pred_model - y_pred_analytical*100, min=0, max=y_pred_analytical*500)) * 0.0001
                    y_pred_batch = y_pred_model + y_pred_analytical
                else:
                    y_pred_batch = self.predict(arch_batch, mapping_batch, access_batch, prob_batch)
                real_loss += self.loss_fn(y_batch, y_pred_batch)
                # weights = None
                # for layer in [0, 3, 6]:
                #     if weights is None:
                #         weights = self.mlps[1][layer].weight.flatten()
                #     else:
                #         weights = torch.cat((weights, self.mlps[1][layer].weight.flatten()))
                # ridge_loss = torch.linalg.norm(weights)
                loss = real_loss
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
            # logger.info(f"Finished training iter {iter}, loss {loss}, real_loss {real_loss} ridge_loss {ridge_loss}") 
            logger.info(f"Finished training iter {iter}, loss {loss}") 
            logger.debug("y_batch: %s, y_pred_batch: %s, mapping_batch: %s, prob_batch: %s, access_batch: %s", 
                         y_batch[-1], y_pred_batch[-1], mapping_batch[-1], prob_batch[-1], access_batch[-1])
            # scheduler.step()
            if (iter+1) % 100 == 0:
                for i, mlp in enumerate(mlps):
                    torch.save(mlp.state_dict(), mlp_paths[i])
                torch.save(optimizer.state_dict(), opt_path)
                if valid_data is not None:
                    for mlp in mlps:
                        mlp.eval()
                    y_pred_valid = self.predict(arch_valid, mapping_valid, access_valid, prob_valid)
                    # y_pred_valid = self.train_predict_noroofline(arch_valid, mapping_valid, access_valid, prob_valid)
                    valid_loss = self.loss_fn(y_valid, y_pred_valid)
                    logger.info("Validation loss %s", valid_loss.item())
                    for mlp in mlps:
                        mlp.train()
        for i, mlp in enumerate(mlps):
            torch.save(mlp.state_dict(), mlp_paths[i])
        torch.save(optimizer.state_dict(), opt_path)

    def predict_analytical(self, hw_config, mapping, access_counts):
        rooflines = self.gen_rooflines(hw_config, mapping, access_counts)
        max_roofline = torch.max(rooflines, dim=1)
        cycle = max_roofline[0].unsqueeze(-1)
        # cycle += (torch.sum(rooflines, dim=1).unsqueeze(-1) - rooflines.gather(1, max_roofline[1].unsqueeze(-1))) * 0.3
        normed_cycle = self.train_data.norm(self.target_key, cycle)
        return normed_cycle

    def predict(self, hw_config, mapping, access_counts, probs):
        # return self.train_predict(mapping, access_counts, probs)
        if self.train_model:
            if self.with_analytical:
                pred_analytical = self.predict_analytical(hw_config, mapping, access_counts)
                pred_model = self.train_predict(hw_config, mapping, access_counts, probs)
                # pred_model = torch.clamp(pred_model, min=pred_analytical*-0.5, max=pred_analytical*2)
                return pred_analytical + pred_model
                # return self.predict_analytical(hw_config, mapping, access_counts) + self.train_predict(hw_config, mapping, access_counts, probs)
            else:
                return self.train_predict_noroofline(hw_config, mapping, probs) + 1e-30
        else:
            return self.predict_analytical(hw_config, mapping, access_counts)
        # return 0.95*self.predict_analytical(hw_config, mapping, access_counts) + 0.05*self.train_predict_noroofline(hw_config, mapping, access_counts, probs)
        # if self.mlps is not None:
        #     return self.train_predict(hw_config, mapping, access_counts, probs)
        # else:
        #     return self.predict_analytical(hw_config, mapping, access_counts)

    def gen_rooflines(self, hw_config, mapping, access_counts):
        hw_config = self.train_data.denorm("arch", hw_config)
        mapping = self.train_data.denorm("mapping", mapping)
        access_counts = self.train_data.denorm("dse.access", access_counts)
        c_sp_idx = self.relevant_mapping_keys.index("mapping.spatial_L1_C")
        k_sp_idx = self.relevant_mapping_keys.index("mapping.spatial_L2_K")
        c_sp = mapping[:, c_sp_idx]
        k_sp = mapping[:, k_sp_idx]

        reg_bw = torch.tensor([[10000] for _ in range(len(mapping))]).to(pytorch_util.device)
        acc_bw = (k_sp * 2).unsqueeze(-1)
        # if len(hw_config.size()) == 1:
        #     acc_r_bw = (hw_config[0]).repeat((len(mapping), 1))
        #     acc_w_bw = (k_sp * 2).unsqueeze(-1)
        # elif len(hw_config.size()) == 2:
        #     acc_r_bw = (hw_config[:, 0]).unsqueeze(-1)
        #     acc_w_bw = (k_sp * 2).unsqueeze(-1)
        # acc_bw = torch.tensor([[10000] for _ in range(len(mapping))]).to(pytorch_util.device)
        if len(hw_config.size()) == 1:
            sp_bw = (hw_config[0] * 2).repeat((len(mapping), 1))
            # sp_r_bw = sp_r_bw = (c_sp * 2).unsqueeze(-1)
            # sp_w_bw = (hw_config[0]).repeat((len(mapping), 1))
        elif len(hw_config.size()) == 2:
            sp_bw = (hw_config[:, 0] * 2).unsqueeze(-1)
            # sp_r_bw = (c_sp * 2).unsqueeze(-1)
            # sp_w_bw = (hw_config[:, 0] * 2).unsqueeze(-1)
            # sp_bw = torch.tensor([[10000] for _ in range(len(mapping))]).to(pytorch_util.device)
        # sp_bw = (c_sp * 2).unsqueeze(-1)
        dram_bw = torch.tensor([[8] for _ in range(len(mapping))]).to(pytorch_util.device)# / 3
        bw = torch.cat(((c_sp * k_sp).unsqueeze(-1), reg_bw, acc_bw, sp_bw, dram_bw), dim=1)
        # bw = torch.cat(((c_sp * k_sp).unsqueeze(-1), reg_bw, acc_r_bw, acc_w_bw, sp_r_bw, sp_w_bw, dram_bw), dim=1)
        if self.with_cache:
            cache_bw = torch.tensor([[64] for _ in range(len(mapping))]).to(pytorch_util.device)
            bw = torch.cat((bw, cache_bw), dim=1)
        rooflines = access_counts / bw
        return rooflines

    def norm_cycle(self, cycle):
        return self.train_data.norm(self.target_key, cycle)
    def denorm_cycle(self, cycle):
        return self.train_data.denorm(self.target_key, cycle)

    def train_predict_noroofline(self, hw_config, mapping, probs):
        if len(hw_config.size()) == 1:
            hw_config = hw_config.repeat((len(mapping), 1))
        y_pred = self.mlps[0](torch.cat((hw_config, mapping[:,self.internal_relevant_idxs], probs), dim=1))
        # y_pred = self.mlps[0](rooflines)
        return y_pred

    def train_predict(self, hw_config, mapping, access_counts, probs):
        if len(hw_config.size()) == 1:
            hw_config = hw_config.repeat((len(mapping), 1))
        rooflines = self.gen_rooflines(hw_config, mapping, access_counts)
        rooflines = rooflines / self.roofline_max
        if self.with_roofline:
            y_pred = self.mlps[0](torch.cat((hw_config, mapping[:,self.internal_relevant_idxs], rooflines, probs), dim=1))
        else:
            y_pred = self.mlps[0](torch.cat((hw_config, mapping[:,self.internal_relevant_idxs], probs), dim=1))
        # y_pred = self.mlps[0](rooflines)
        return y_pred

        # intermed_pred = self.mlps[0](torch.cat((mapping, probs), dim=1))
        # y_pred = self.mlps[1](torch.cat((intermed_pred, access_counts), dim=1))
        # y_pred = self.mlps[0](torch.cat((access_counts, mapping, probs), dim=1))
        # num_mem_lvls = 4
        # num_dims = 7
        c_sp_idx = self.relevant_mapping_keys.index("mapping.spatial_L1_C")
        k_sp_idx = self.relevant_mapping_keys.index("mapping.spatial_L2_K")
        c_sp_max = self.train_data.creator.stats["mapping.spatial_L1_C_max"]
        k_sp_max = self.train_data.creator.stats["mapping.spatial_L2_K_max"]
        c_sp_min = self.train_data.creator.stats["mapping.spatial_L1_C_min"]
        k_sp_min = self.train_data.creator.stats["mapping.spatial_L2_K_min"]
        c_sp = (mapping[:, c_sp_idx] * (c_sp_max - c_sp_min)) + c_sp_min
        k_sp = (mapping[:, k_sp_idx] * (k_sp_max - k_sp_min)) + k_sp_min
        spatial_prod = c_sp * k_sp
        spatial_prod = spatial_prod.unsqueeze(-1)
        macs = access_counts[:, 0].unsqueeze(-1)
        compute_cycles = macs / spatial_prod
        dram_accesses = access_counts[:, -1].unsqueeze(-1)
        # y_pred = self.mlps[0](torch.cat((access_counts,spatial_prod,mapping), dim=1))
        y_pred = self.mlps[0](torch.cat((mapping, compute_cycles, dram_accesses, probs), dim=1))
        return y_pred
        # return self.mlp(access_counts)
        # return self.mlp(torch.cat((mapping, access_counts, probs), dim=1))
        # return self.mlp(torch.cat((mapping, probs), dim=1))

    def freeze(self):
        if not self.mlps:
            return
        for mlp in self.mlps:
            mlp.eval()
            for p in mlp.parameters():
                p.requires_grad = False

    def unfreeze(self):
        if not self.mlps:
            return
        for mlp in self.mlps:
            mlp.train()
            for p in mlp.parameters():
                p.requires_grad = True

    def test(self, test_data: DlaDataset, num_worst_points=10, gpu_id=0, num_unplot_points=0):
        stats = test_data.creator.stats
        arch_keys = utils.keys_by_type(test_data.df, "arch", scalar_only=True)
        mapping_keys = self.relevant_mapping_keys
        all_mapping_keys = utils.keys_by_type(test_data.df, "mapping", scalar_only=True)
        prob_keys = utils.keys_by_type(test_data.df, "prob", scalar_only=True)
        access_keys = utils.keys_by_type(test_data.df, "dse.access", scalar_only=True)
        y_keys = [self.target_key]

        arch_test = test_data.df[arch_keys].to_numpy()
        mapping_test = test_data.df[mapping_keys].to_numpy()
        all_mapping_test = test_data.df[all_mapping_keys].to_numpy()
        prob_test = test_data.df[prob_keys].to_numpy()
        y_test = test_data.df[y_keys].to_numpy()
        
        access_test = test_data.df[access_keys]
        access_test_denormed = access_test.copy()
        for access_key in access_keys:
            mean = stats[access_key + "_max"]
            logger.debug("%s: %s", access_key + "_max", mean)
            if mean != 0:
                access_test_denormed[access_key] = access_test_denormed[access_key] * mean
        access_test = access_test.to_numpy()
        access_test_denormed = access_test_denormed.to_numpy()

        # remove worst points
        mask = None
        if num_unplot_points > 0:
            logger.info("Removing %s worst testing points", num_unplot_points)
            mask = np.argpartition(np.squeeze(y_test), kth=-num_unplot_points)[:-num_unplot_points]
            arch_test = arch_test[mask]
            mapping_test = mapping_test[mask]
            all_mapping_test = all_mapping_test[mask]
            prob_test = prob_test[mask]
            access_test = access_test[mask]
            access_test_denormed = access_test_denormed[mask]
            y_test = y_test[mask]

        arch_test = pytorch_util.from_numpy(arch_test)
        mapping_test = pytorch_util.from_numpy(mapping_test)
        all_mapping_test = pytorch_util.from_numpy(all_mapping_test)
        prob_test = pytorch_util.from_numpy(prob_test)
        access_test = pytorch_util.from_numpy(access_test)
        access_test_denormed = torch.tensor(access_test_denormed)
        y_test = pytorch_util.from_numpy(y_test)
        y_test_denormed = test_data.denorm(self.target_key, y_test)
        test_dataset = pytorch_util.X_y_dataset(arch_test, mapping_test, prob_test, y_test, access_test)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40000)
        y_pred = None
        with torch.no_grad():
            # for mlp in self.mlps:
            #     mlp.eval()
            for arch_batch, mapping_batch, prob_batch, y_batch, access_batch in test_data_loader:
                y_pred_batch = self.predict(arch_batch, mapping_batch, access_batch, prob_batch)
                if y_pred is None:
                    y_pred = y_pred_batch
                else:
                    y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
        
        losses = []
        for col in range(y_test.size(1)):
            col_loss = self.loss_fn(y_test[:,col], y_pred[:,col])
            logger.info("Col %s loss: %s", y_keys[col], col_loss)
            losses.append(col_loss)

            loss_fn_no_red = torch.nn.MSELoss(reduction="none")
            if num_worst_points > 0:
                col_loss_no_red = loss_fn_no_red(y_test[:,col], y_pred[:,col])
                col_worst_losses, col_worst_idxs = col_loss_no_red.topk(num_worst_points)
                # logger.info("Col %s worst points idxs: %s", y_keys[col], col_worst_idxs)
                # logger.info("Col %s worst points losses: %s", y_keys[col], col_worst_losses)
                # logger.info("Col %s worst points mapping-vals: %s", y_keys[col], mapping_test[col_worst_idxs])
                # logger.info("Col %s worst points mapping-vals (denormed): %s", y_keys[col], test_data.denorm("mapping", all_mapping_test[col_worst_idxs]))
                # logger.info("Col %s worst points access-vals: %s", y_keys[col], access_test[col_worst_idxs])
                # logger.info("Col %s worst points access-vals (denormed): %s", y_keys[col], access_test_denormed[col_worst_idxs])
                # logger.info("Col %s worst points Y-vals: %s", y_keys[col], y_test[:,col][col_worst_idxs])
                # logger.info("Col %s worst points Y-vals (denormed): %s", y_keys[col], y_test_denormed[:,col][col_worst_idxs])
                # logger.info("Col %s worst points Y-preds: %s", y_keys[col], y_pred[:,col][col_worst_idxs])
                # logger.info("Col %s worst points Y-preds (denormed): %s", y_keys[col], test_data.denorm(self.target_key, y_pred)[:,col][col_worst_idxs])
                # if mask:
                #     logger.info("Col %s worst points data: %s", y_keys[col], test_data.df.iloc[mask].iloc[pytorch_util.to_numpy(col_worst_idxs)].to_string())
                # else:
                #     logger.info("Col %s worst points data: %s", y_keys[col], test_data.df.iloc[pytorch_util.to_numpy(col_worst_idxs)].to_string())

        y_test = test_data.denorm(self.target_key, pytorch_util.to_numpy(y_test))
        y_pred = test_data.denorm(self.target_key, pytorch_util.to_numpy(y_pred))
        return y_test, y_pred, losses
