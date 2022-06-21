import os
import random
import time

import audtorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataloador.Exvo_A import CachedDataset, CCCLoss
from models.vmodels import ResNet14
from train_cfg import TrainConfig
from utils.metrics import EvalMetrics


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    local = False
    if machine == '[YOUR MACHINE NAME]':
        prefix = '[YOUR LOCAL PATH PREFIX]'
        local = True
    return prefix, local


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, current_model, current_epoch, marker, timestamp):
    save_path = os.path.join(save_path, timestamp)
    save_to = os.path.join(save_path, '{}_{}.pkl'.format(marker, current_epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(current_model, save_to)
    print('<== Model is saved to {}'.format(save_to))

def load_model(model_path, model_name):
    model = torch.load(os.path.join(model_path, model_name))
    return model


def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))


def print_flags(cfg):
    print('--------------------------- Flags -----------------------------')
    for flag in cfg.asdic():
        print('{} : {}'.format(flag, getattr(cfg, flag)))


def report_metrics(pred_aggregate_, gold_aggregate_):
    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))

    print(classification_report(gold_aggregate_, pred_aggregate_))
    print(confusion_matrix(gold_aggregate_, pred_aggregate_))


def create_ds(cfg):
    target_column = label_names
    df = pd.read_csv(os.path.join(cfg.data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df_train = df.loc[df['Split'] == 'Train']
    df_dev = df.loc[df['Split'] == 'Val']
    df_test = df.loc[df['Split'] == 'Val']

    features = pd.read_csv(cfg.features_path).set_index('file')
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(cfg.features_path), os.path.basename(x)))

    db_args = {
        'features': features,
        'target_column': target_column,
        'transform': audtorch.transforms.RandomCrop(250, axis=-2),
    }

    ds_tr = CachedDataset(df_train, **db_args)
    ds_ev = CachedDataset(df_dev, **db_args)

    ds_len = len(ds_tr)
    indices = torch.randperm(ds_len)[:cfg.tr_sm]
    ds_tr_ = torch.utils.data.Subset(ds_tr, indices)
    print('# tr samples: {}, percentage: {:.4}'.format(len(ds_tr_), len(ds_tr_) / ds_len))
    return ds_tr_, ds_ev


def create_tr_dl(dataset, batch_size=4):
    dl_tr = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return dl_tr


def create_val_dl(dataset, batch_size=4):
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    return val_dl


def create_dl(cfg):
    ds_tr, ds_val = create_ds(cfg)
    dl_tr = create_tr_dl(ds_tr, cfg.tr_bs)
    dl_val = create_val_dl(ds_val, cfg.val_bs)
    return dl_tr, dl_val


def training_setting(model, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = CCCLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.9,
        patience=5
    )
    return optimizer, loss_fn, scheduler


def train(dl, optimizer, loss_fn, epoch, log_freq=10):
    losses = 0.
    counter = 1
    tmp_losses = 0.
    tmp_counter = 0
    pred = []
    gt = []
    ccc_result = []
    model.train()
    for idx, (batch, label) in enumerate(dl):
        batch = batch.to(device)
        label = label.to(device)

        out = model(batch)
        del batch
        optimizer.zero_grad()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        counter += 1

        tmp_losses += loss.item()
        tmp_counter += 1

        if idx % log_freq == 0:
            print('   [{}/{}] Train loss: {}'.format(idx, len(dl), tmp_losses / tmp_counter))

            tmp_losses = 0.
            tmp_counter = 0

        pred.append(out.detach().cpu().numpy())
        gt.append(label.detach().cpu().numpy())

    pred = np.stack(pred)
    gt = np.stack(gt)

    for idx in range(10):
        ccc = EvalMetrics.CCC(gt[:, :, idx].flatten(), pred[:, :, idx].flatten())
        ccc_result.append(ccc)


    print('##>[{}]Train Results: {:4f}'.format(epoch, np.mean(ccc_result)))


def eval(dl, loss_fn, log_freq=6000):
    losses = 0.
    counter = 1

    pred = []
    gt = []
    ccc_result = []
    model.eval()
    for idx, (batch, label) in enumerate(dl):
        batch = batch.to(device)
        label = label.to(device)

        out = model(batch)
        del batch
        loss = loss_fn(out, label)

        losses += loss.item()
        counter += 1

        if idx % log_freq == 0:
            print('   [{}/{}]'.format(idx, len(dl)))

        pred.append(out.detach().cpu().numpy())
        gt.append(label.detach().cpu().numpy())

    pred = np.stack(pred)
    gt = np.stack(gt)

    for idx in range(10):
        ccc = EvalMetrics.CCC(gt[:, :, idx].flatten(), pred[:, :, idx].flatten())
        ccc_result.append(ccc)

    print('==>[{}]Val Results: {:4f}'.format(epoch, np.mean(ccc_result)))
    print('-' * 64)


if __name__ == "__main__":
    label_names = [
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]
    machine, local = get_path_prefix()
    tr_cfg = TrainConfig()
    print_flags(tr_cfg)

    save_path = machine + tr_cfg.save2
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if not local:
        setup_seed(tr_cfg.random_seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not tr_cfg.resume:
        model = ResNet14(num_classes=10)
        # model = Cnn10(10)

    model.to(device)
    print_nn(model)
    print('-' * 64)

    tr_dl, val_dl = create_dl(tr_cfg)
    optimizer, loss_fn, scheduler = training_setting(model, tr_cfg.lr)

    for epoch in range(1, tr_cfg.epoches + 1):
        train(tr_dl, optimizer, loss_fn, epoch, tr_cfg.log_freq)
        if epoch % tr_cfg.save_every == 0:
            save_model(save_path, model, epoch, tr_cfg.md_name, timestamp)

        if epoch % tr_cfg.val_every == 0:
            eval(val_dl, loss_fn)
