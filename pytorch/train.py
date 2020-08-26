import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from contextlib import contextmanager
from time import time
import utils as ut


def train(model, optimizer, loss_fn, dls, num_epochs, log_dir, device):
    train_loader = dls['train']
    val_loader = dls['val']
    ntrain = len(train_loader.dataset)
    nval = len(val_loader.dataset)
    writer = SummaryWriter(log_dir=log_dir)

    loss_tr = []
    loss_val = []
    model = model.to(device)
    for ep in np.arange(num_epochs):
        start = time()
        running_loss_tr = 0.0
        model.train()
        for x_tr_b, y_tr_b, idx_tr_b in train_loader:
            # Forward pass
            x_tr_b = x_tr_b.to(device)
            y_tr_b = y_tr_b.to(device)
            optimizer.zero_grad()

            score_tr_b = model(x_tr_b)
            loss_tr_b = loss_fn(score_tr_b, y_tr_b)
            loss_tr_mean_b = torch.mean(loss_tr_b)

            # Backward and optimize
            loss_tr_mean_b.backward()
            optimizer.step()

            running_loss_tr += torch.sum(loss_tr_b)

        loss_tr = running_loss_tr / ntrain
        # loss_tr.append(loss_tr)

        ##### Model Validation ##########
        y_val = []
        score_val = []
        yhat_val = []
        best_val_loss = np.inf
        with torch.no_grad():
            model.eval()
            running_loss_val = 0
            for x_val_b, y_val_b, idx_te_b in val_loader:
                x_val_b = x_val_b.to(device)
                y_val_b = y_val_b.to(device)

                # Forward pass
                score_val_b = model(x_val_b)  # logits
                _, yhat_val_b = torch.max(score_val_b, 1)
                loss_val_b = loss_fn(score_val_b, y_val_b)

                # book keeping at batch level
                running_loss_val += torch.sum(loss_val_b)
                y_val.append(y_val_b)
                score_val.append(score_val_b)
                yhat_val.append(yhat_val_b)

            loss_val = running_loss_val / nval
            is_best = loss_val < best_val_loss

            # predictions and  metrics
            y_val = torch.cat(y_val).cpu().numpy()
            score_val = torch.cat(score_val).cpu().numpy()
            yhat_val = torch.cat(yhat_val).cpu().numpy()

            acc = accuracy_score(y_val, yhat_val)
            f1 = f1_score(y_val, yhat_val)
            pre = precision_score(y_val, yhat_val)
            recall = recall_score(y_val, yhat_val)
            auc = roc_auc_score(y_val, score_val[:, 1])

            if is_best:
                ut.save_ckpt({'epoch': ep + 1,
                              'state_dict': model.state_dict(),
                              'optim_dict': optimizer.state_dict()},
                             {'score_val': score_val,
                                 'loss_val': loss_val,
                                 'y_val': y_val,
                                 'yhat_val': yhat_val},
                             {'epoch': ep+1,
                                 'acc': acc,
                                 'f1': f1,
                                 'pre': pre,
                                 'recall': recall,
                                 'auc': auc},
                             is_best=is_best,
                             ckpt_dir=log_dir)

            # logging
            writer.add_scalar('train/loss', loss_tr, ep)
            writer.add_scalar('val/loss', loss_val, ep)
            writer.add_scalar('val/metrics/acc', acc, ep)
            writer.add_scalar('val/metrics/auc', auc, ep)
            writer.add_scalar('val/metrics/f1', f1, ep)
            writer.add_scalar('val/metrics/pre', pre, ep)
            writer.add_scalar('val/metrics/recall', recall, ep)
            et = time() - start

            print(f"ep:{ep}|et{et:.3f}|tr: {loss_tr:.5f}|loss: {loss_val:.5f}|acc:{acc:.5f}" +
                  f"|f1:{f1:.5f}|auc:{auc:.5f}|pre:{pre:.5f}|recall:{recall:.5f}")
