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
from rumex_dataset import RumexDataset, train_loader, test_loader
from rumex_model import RumexNet
from torch_lr_finder import LRFinder


def trainer(cfg):
    gpu = cfg.gpu & torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    log_dir = cfg['model_name']+'_logs'
    dstr = RumexDataset(cfg['data_dir']+'valid/', train_flag=True)
    dsva = RumexDataset(cfg['data_dir']+'train/', train_flag=False)

    dltr = train_loader(dstr, cfg['bs'])
    dlva = test_loader(dsva, 2*cfg['bs'])

    model = RumexNet(cfg['model_name'])
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['lr'],
                                 weight_decay=1e-2)

    # writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = np.inf
    for ep in np.arange(cfg['num_epochs']):
        start = time()

        #### fit model ##########
        loss = train(model, dltr, optimizer, loss_fn, device)

        ##### Model Validation ##########
        predictions, metrics = validate(model, dlva, loss_fn, device)

        ##### checkpoint saving and logging ##########
    if (metrics['loss'] < best_val_loss) & (metrics['acc'] > best_val_acc):
        best_val_loss = metrics['loss']
        best_val_acc = metrics["acc"]
        ckpt_dict = {'ep': ep,
                     'state_dict': model.state_dict(),
                     'optim_dict': optimizer.state_dict(),
                     'predictions': predictions,
                     'metrics': metrics}

        ut.save_ckpt(ckpt_dict, log_dir)

        # # tensorboad logging
        # writer.add_scalar('train/loss', loss, ep)
        # for key in metrics.keys():
        #     name = 'val/'+key
        #     writer.add_scalar(name, metrics[key], ep)
        # et = time() - start

        print(f"ep:{ep} et:{et:0f}|loss_tr:{loss:.5f}|loss: {metrics['loss']:.5f}" +
              f"|acc:{metrics['acc']:.5f}|re:{metrics['pre']:.5f}" +
              f"|pre:{metrics['recall']:.5f}|f1:{metrics['f1']:.5f}")


def train(model, dl, optimizer, scheduler, loss_fn, device):
    model.to(device)
    model.train()
    running_loss = 0.0
    for xb, yb, _, _ in dl:
        # Forward pass
        xb = xb.to(device)
        yb = yb.to(device)
        scoreb = model(xb)
        # loss of each elem in batch
        lossb = loss_fn(scoreb, yb)
        lossb_mean = torch.mean(lossb)

        # Backward and optimize
        lossb_mean.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        running_loss += torch.sum(lossb)

    loss = running_loss / len(dl.dataset)
    return loss


def validate(model, dl, loss_fn, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        fname = []
        score = []
        loss = []
        y = []
        yhat = []
        for xb, yb, _, fnameb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            scoreb = model(xb)  # logits
            _, yhatb = torch.max(scoreb, 1)
            lossb = loss_fn(scoreb, yb)

            # book keeping at batch level
            score.append(scoreb)
            yhat.append(yhatb)
            loss.append(lossb)
            fname.append(fnameb)
            y.append(yb)

        score = torch.cat(score)
        loss = torch.cat(loss)
        y = torch.cat(y)
        yhat = torch.cat(yhat)

        # flatten list of tuples
        fname = [y for x in fname for y in x]

        # predictions and metrics
        # _, yhat = torch.max(score, 1)
        loss_mean = torch.mean(loss)

        metrics = ut.compute_metrics(y, yhat, score[:, 1])
        metrics['loss'] = loss_mean

        predictions = {
            # 'idx': idx,
            'fname': fname,
            'y': y,
            'yhat': yhat,
            'score': score[:, 1],
            'loss': loss}

    return predictions, metrics
