import torch
# import logging
# import argparse
import numpy as np
import utils as ut
from time import time
from torch import optim
# from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self,
                 max_ep=10,
                 patience=5,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 loss_fn=None,
                 dltr=None,
                 dlva=None,
                 dlte=None,
                 bs=None,
                 device=None,
                 log_dir="logs_temp",
                 exp_name="temp"):

        # TODO: add comments to the code
        # TODO: set deraults to reasonable values
        # TODO: add code for best LR finder

        self.device = device
        self.exp_name = exp_name
        self.log_dir = log_dir

        # # self.writer = SummaryWriter(log_dir=log_dir)
        # logging.info(eq50 + 'training started' + eq50)
        # logging.info(cfg)

        # model, optmizer and loss_fn
        self.init_model = model
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.loss_fn = loss_fn

        # dataset and data loader
        self.bs = bs
        self.dltr = dltr
        self.dlva = dlva
        self.dlte = dlte

        # training state
        # self.lr = cfg.lr
        self.training = False
        self.testing = False
        self.validating = False
        self.current_ep = 0
        self.current_losstr = np.inf
        self.current_lossva = np.inf
        self.current_losste = np.inf

        self.es_ep = 0
        self.patience = patience
        self.patience_counter = 0

        self.best_lossva = np.inf
        self.max_ep = max_ep

        self.start_time = None
        self.end_time = None

        # record history; do not include score and predictions from train set
        self.losstr_history = []
        self.lossva_history = []

        # record predictions and metrics
        # training metrics not included as its mememory intensive (TODO?)
        self.scoreva = None
        self.yva = None
        self.yhatva = None
        self.metricsva = None

        self.scorete = None
        self.yte = None
        self.yhatte = None
        self.metricste = None
        self.str = "="*25

    def fit(self):
        for ep in np.arange(self.max_ep):
            start_time = time()
            self.current_ep = ep
            self.train(self.dltr)  # train model
            end_time = time()
            et = end_time - start_time

            print(f"ep:{self.current_ep}/{self.max_ep}|"
                  + f"et:{et:.0f}|"
                  + f"losstr:{self.current_losstr:.5f}|")

        self.test(self.dlte, split="test")
        print(f"{self.str}test performance{self.str}")
        print(f"losste:{self.current_losste:.5f}|"
              + f"accte:{self.metricste['acc']:.5f}|"
              + f"aucte:{self.metricste['auc']:.5f}|"
              + f"f1te:{self.metricste['f1']:.5f}|"
              + f"p1te:{self.metricste['p1']:.5f}|"
              + f"r1te:{self.metricste['r1']:.5f}|"
              )

    def train(self, dl):
        self.model.train()
        running_loss = 0.0
        for xb, yb, _ in dl:
            # Forward pass
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            scoreb = self.model(xb)
            lossb = self.loss_fn(scoreb, yb)
            # Backward and optimize
            lossb_mean = torch.mean(lossb)
            lossb_mean.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # scheduler.step()
            running_loss += torch.sum(lossb)

        self.current_losstr = running_loss / len(dl.dataset)

        # self.current_losstr = torch.mean(torch.cat(losstr))
        self.losstr_history.append(self.current_losstr.item())

    def test(self, dl, split):
        ##### test model ##########
        self.model.eval()
        score = []
        y = []
        yhat = []
        loss = []
        fname = []
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb, fnameb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Forward pass
                scoreb = self.model(xb)  # logits
                lossb = self.loss_fn(scoreb, yb)
                _, yhatb = torch.max(scoreb, 1)

                # book keeping at batch level
                running_loss += torch.sum(lossb)

                # self.fnameva.append(fnameb)
                score.append(scoreb)
                yhat.append(yhatb)
                y.append(yb)
                fname.append(fnameb)

            current_loss = running_loss / len(dl.dataset)

            # flatten lists
            fname = [b for a in fname for b in a]  # flatten list of tuples
            score = torch.cat(score)
            y = torch.cat(y)
            yhat = torch.cat(yhat)

            # compute metrics
            metrics = ut.compute_metrics(y, yhat, score[:, 1])

            if split == "valid":
                self.current_lossva = current_loss
                self.lossva_history.append(self.current_lossva.item())
                self.scoreva = score
                self.yva = y
                self.yhatva = yhat
                self.fnameva = fname
                self.metricsva = metrics

            else:
                self.current_losste = current_loss
                self.scorete = score
                self.yte = y
                self.yhatte = yhat
                self.fnamete = fname
                self.metricste = metrics
