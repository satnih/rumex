import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
import utils as ut
from rumex_dataset import RumexDataset
from rumex_model import RumexNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg):
    log_dir = cfg['model_name']+'_logs'
    dstr = RumexDataset(cfg['data_dir']+'train/', train_flag=True)
    dsva = RumexDataset(cfg['data_dir']+'valid/', train_flag=False)

    dltr = dstr.make_data_loader(cfg['bs'])
    dlva = dsva.make_data_loader(cfg['bs'])

    model = RumexNet(cfg['model_name'])
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)
    best_val_loss = np.inf
    for ep in np.arange(cfg['num_epochs']):
        start = time()

        #### fit model ##########
        model.train()
        loss = fit(model, dltr, optimizer, loss_fn)

        ##### Model Validation ##########
        model.eval()
        predictions, metrics = validate(model, dlva, loss_fn)

        ##### checkpoint saving and logging ##########
        is_best = metrics['loss'] < best_val_loss
        if ep % cfg['log_freq'] == 0:
            model_state = {'epoch': ep + 1,
                           'state_dict': model.state_dict(),
                           'optim_dict': optimizer.state_dict()}
            ut.save_ckpt(model, predictions, metrics, is_best, log_dir)

        # tensorboad logging
        writer.add_scalar('train/loss', loss, ep)
        for key in metrics.keys():
            name = 'val/'+key
            writer.add_scalar(name, metrics[key], ep)
        et = time() - start

        print(f"ep:{ep} et:{et:0f}|loss_tr:{loss:.5f}|loss_te: {metrics['loss']:.5f}" +
              f"|acc:{metrics['acc']:.5f}|re:{metrics['pre']:.5f}|pre:{metrics['recall']:.5f}" +
              f"|f1:{metrics['f1']:.5f}|auc:{metrics['auc']:.5f}")


def fit(model, dl, optimizer, loss_fn):
    running_loss = 0.0
    for xb, yb, _, _ in dl:
        # Forward pass
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()

        scoreb = model(xb)
        # loss of each elem in batch
        lossb = loss_fn(scoreb, yb)
        lossb_mean = torch.mean(lossb)

        # Backward and optimize
        lossb_mean.backward()
        optimizer.step()

        running_loss += torch.sum(lossb)

    loss = running_loss / len(dl.dataset)
    return loss


def validate(model, dl, loss_fn):
    with torch.no_grad():
        idx = []
        fname = []
        score = []
        loss = []
        y = []
        for xb, yb, _, fnameb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            scoreb = model(xb)  # logits
            lossb = loss_fn(scoreb, yb)

            # book keeping at batch level
            score.append(scoreb)
            loss.append(lossb)
            # idx.append(idxb)
            fname.append(fnameb)
            y.append(yb)

        score = torch.cat(score)
        loss = torch.cat(loss)
        # idx = torch.cat(idx)
        y = torch.cat(y)

        # flatten list of tuples
        fname = [y for x in fname for y in x]

        # predictions and metrics
        _, yhat = torch.max(score, 1)
        loss_mean = torch.mean(loss)

        metrics = compute_metrics(y, yhat, score[:, 1])
        metrics['loss'] = loss_mean

        predictions = {
            # 'idx': idx,
            'fname': fname,
            'y': y,
            'yhat': yhat,
            'score': score[:, 1],
            'loss': loss}

    return predictions, metrics


def compute_metrics(y, yhat, score):
    tp = torch.sum((y == 1) & (yhat == 1)).float()
    fp = torch.sum((y == 1) & (yhat == 0)).float()
    tn = torch.sum((y == 0) & (yhat == 0)).float()
    fn = torch.sum((y == 0) & (yhat == 1)).float()
    acc = (tp+tn)/len(y)

    recall = tp/(tp + fn)  # predicted pos/condition pos
    precision = tn/(tn + fp)  # predicted neg/condition neg
    f1 = 2*precision*recall/(precision+recall)
    auc = roc_auc_score(y.cpu().numpy(), score.cpu().numpy())
    metrics = {
        'acc': acc,
        'f1': f1,
        'pre': precision,
        'recall': recall,
        'auc': auc}
    return metrics
