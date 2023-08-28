import pathlib
import traceback

import numpy as np
import torch

from dataset.common import logger, utils
from dataset.dse import pytorch_util, DlaDataset

loss_fn = torch.nn.MSELoss()

def train_linear(train_data: DlaDataset, x_key_types, y_key, num_iters=200, output_dir=pathlib.Path(""), save_str="",gpu_id=0,
              continue_training=False):
    return train_mlp(train_data, x_key_types, [y_key], hidden_layer_sizes=(), num_iters=num_iters, output_dir=output_dir, save_str=save_str,
                     gpu_id=gpu_id, continue_training=continue_training, dropout=0, interp_points=0)

def train_mlp(train_data: DlaDataset, x_key_types, y_keys, hidden_layer_sizes=(128, 64, 32), num_iters=200, output_dir=pathlib.Path(""), save_str="",gpu_id=0,
              continue_training=False, dropout=0.2, interp_points=0):
    mlp_path = pathlib.Path(output_dir).resolve() / f"mlp_{save_str}.pt"
    opt_path = pathlib.Path(output_dir).resolve() / f"mlp_opt_{save_str}.pt"
    sch_path = pathlib.Path(output_dir).resolve() / f"mlp_sch_{save_str}.pt"

    train_df = train_data.df
    
    x_keys = []
    for key_type in x_key_types:
        type_keys = utils.keys_by_type(train_df, key_type, scalar_only=True)
        x_keys.extend(type_keys)

    pytorch_util.init_gpu(gpu_id=gpu_id)
    mlp = pytorch_util.build_mlp(
        input_size=len(x_keys),
        output_size=len(y_keys),
        n_layers=len(hidden_layer_sizes),
        size=hidden_layer_sizes,
        activation="relu",
        dropout=dropout,
    )
    mlp.to(pytorch_util.device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    try:
        mlp.load_state_dict(torch.load(mlp_path))
        optimizer.load_state_dict(torch.load(opt_path))
        logger.info("Loaded existing MLP and optimizer from %s and %s", mlp_path, opt_path)
        if not continue_training:
            return mlp
    except:
        print(traceback.format_exc())
        pass
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 5e-3, cycle_momentum=False, mode="triangular2")
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2)
    # try:
    #     scheduler.load_state_dict(torch.load(sch_path))
    # except:
    #     pass

    X_train = train_df[x_keys].to_numpy()
    y_train = train_df[y_keys].to_numpy()

    if interp_points > 0:
        logger.info("Adding %s interpolation points", interp_points)
        idx_pairs = np.random.randint(0, len(X_train), size=(2, interp_points*20))
        first_points = X_train[idx_pairs[0]]
        second_points = X_train[idx_pairs[1]]
        print(first_points.shape)
        norms = np.linalg.norm(first_points - second_points, axis=1)
        print(norms.shape)
        mask = np.argpartition(norms, kth=interp_points)[:interp_points]
        first_X = first_points[mask]
        second_X = second_points[mask]
        print(np.median(norms), np.median(np.linalg.norm(first_X - second_X, axis=1)))
        first_y = y_train[idx_pairs[0]][mask]
        second_y = y_train[idx_pairs[1]][mask]
        ratios = np.expand_dims(np.random.rand(interp_points), -1)
        X_train_interp = ratios * first_X + (1-ratios) * second_X
        y_train_interp = ratios * first_y + (1-ratios) * second_y
        X_train = np.vstack((X_train, X_train_interp))
        y_train = np.vstack((y_train, y_train_interp))

    logger.info("Training prediction of target(s) %s with inputs %s for %s iterations, %s data points",
                y_keys, x_key_types, num_iters, len(X_train))
    X_train = pytorch_util.from_numpy(X_train)
    y_train = pytorch_util.from_numpy(y_train)
    train_dataset = pytorch_util.X_y_dataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40000)
    mlp.train()
    for iter in range(num_iters):
        for X_batch, y_batch in train_data_loader:
            y_train_pred_batch = mlp(X_batch)
            loss = None
            for output in range(y_batch.size(1)):
                this_output_loss = loss_fn(y_train_pred_batch[:,output], y_batch[:,output])
                if loss is None:
                    loss = this_output_loss
                else:
                    loss += this_output_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"Finished training iter {iter}, loss {loss}")
        if scheduler:
            scheduler.step()
        if (iter+1) % 20 == 0:
            torch.save(mlp.state_dict(), mlp_path)
            torch.save(optimizer.state_dict(), opt_path)
            if scheduler:
                torch.save(scheduler.state_dict(), sch_path)
    torch.save(mlp.state_dict(), mlp_path)
    torch.save(optimizer.state_dict(), opt_path)
    if scheduler:
        torch.save(scheduler.state_dict(), sch_path)
    return mlp

def test_model(model, test_data: DlaDataset, x_key_types, y_keys, num_worst_points=10):
    test_df = test_data.df
    x_keys = []
    for key_type in x_key_types:
        type_keys = utils.keys_by_type(test_df, key_type, scalar_only=True)
        x_keys.extend(type_keys)

    X_test = test_df[x_keys]
    y_test = test_df[y_keys]

    X_test = pytorch_util.from_numpy(X_test.to_numpy())
    y_test = pytorch_util.from_numpy(y_test.to_numpy())
    test_dataset = pytorch_util.X_y_dataset(X_test, y_test)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20000)
    y_pred = pytorch_util.from_numpy(np.array([]))
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_data_loader:
            y_batch_pred = model(X_batch)
            y_pred = torch.cat((y_pred, y_batch_pred))
    
    losses = []
    for col in range(y_test.size(1)):
        col_loss = loss_fn(y_test[:,col], y_pred[:,col])
        logger.info("Col %s loss: %s", col_loss)
        losses.append(col_loss)

        loss_fn_no_red = torch.nn.MSELoss(reduction="none")
        if num_worst_points > 0:
            col_loss_no_red = loss_fn_no_red(y_test[:,col], y_pred[:,col])
            col_worst_losses, col_worst_idxs = col_loss_no_red.topk(num_worst_points)
            logger.info("Col %s worst points idxs: %s", y_keys[col], col_worst_idxs)
            logger.info("Col %s worst points losses: %s", y_keys[col], col_worst_losses)
            logger.info("Col %s worst points X-vals: %s", y_keys[col], X_test[col_worst_idxs])
            logger.info("Col %s worst points X-vals (denormed): %s", y_keys[col], test_data.denorm(x_key_types, X_test[col_worst_idxs]))
            logger.info("Col %s worst points Y-vals: %s", y_keys[col], y_test[:,col][col_worst_idxs])
            logger.info("Col %s worst points Y-preds: %s", y_keys[col], y_pred[:,col][col_worst_idxs])
            logger.info("Col %s worst points data: %s", y_keys[col], test_data.df.iloc[pytorch_util.to_numpy(col_worst_idxs)].to_string())

    y_test = pytorch_util.to_numpy(y_test)
    y_pred = pytorch_util.to_numpy(y_pred)
    return y_test, y_pred, losses
