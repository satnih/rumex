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
        self.scoresva = []
        self.yva = []
        self.yhatva = []
        self.metricsva = None

        self.scoreste = []
        self.yte = []
        self.yhatte = []
        self.metricste = None
        self.str = "="*25

    def fit(self):
        self.start_time = time()
        for ep in np.arange(self.max_ep):
            self.current_ep = ep
            self.train(self.dltr)  # train model
            self.test(self.dlva, split="valid")  # test model on validation set

            ##### early stopping ##########
            # from deep larning book
            if self.current_ep == 0:
                self.es_ep = self.current_ep  # i in  book
                self.patience_counter = 0  # j in  book
                self.best_lossva = np.inf  # nu in book
            elif self.current_ep > 10:
                if self.current_lossva < self.best_lossva:
                    # update early stopping variables
                    self.best_lossva = self.current_lossva
                    self.best_ep = self.current_ep  # es epoch (i* in book)
                    self.patience_counter = 0  # reset patience counter
                    # self.save_checkpoint(self.best_ep)
                else:
                    self.patience_counter += 1

            if self.patience_counter > self.patience:
                print(f"stopping early: best at ep{self.best_ep}{self.str}")
                break

            self.end_time = time()
            et = self.end_time - self.start_time

            print(f"ep:{self.current_ep}/{self.max_ep}|"
                  + f"et:{et:.0f}|"
                  + f"losstr:{self.current_losstr:.5f}|"
                  + f"lossva:{self.current_lossva:.5f}|"
                  + f"accva:{self.metricsva['acc']:.5f}|"
                  + f"f1va:{self.metricsva['f1']:.5f}|"
                  + f"aucva:{self.metricsva['auc']:.5f}"
                  )

        print(f"{self.str}retraining using all data{self.str}")

        self.model = self.init_model
        ds = torch.utils.data.ConcatDataset([self.dltr.dataset,
                                             self.dlva.dataset])
        dl = ut.train_loader(ds, self.bs)
        for ep in np.arange(self.best_ep):
            self.current_ep = ep
            self.train(dl)
            print(f"ep:{self.current_ep}/{self.best_ep}|"
                  + f"losstr:{self.current_losstr:.5f}")

        self.test(self.dlte, split="test")

        print(f"{self.str}test performance{self.str}")
        print(f"losste:{self.current_losste:.5f}|"
              + f"losste:{self.current_losste:.5f}|"
              + f"accte:{self.metricste['acc']:.5f}|"
              + f"f1te:{self.metricste['f1']:.5f}|"
              + f"aucte:{self.metricste['auc']:.5f}")

    def train(self, dl):
        self.model.train()
        running_loss = 0.0
        for xb, yb, _, _ in dl:
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
        scores = []
        y = []
        yhat = []
        loss = []
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb, _, _ in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Forward pass
                scoreb = self.model(xb)  # logits
                lossb = self.loss_fn(scoreb, yb)
                _, yhatb = torch.max(scoreb, 1)

                # book keeping at batch level
                running_loss += torch.sum(lossb)

                # self.fnameva.append(fnameb)
                scores.append(scoreb)
                yhat.append(yhatb)
                y.append(yb)

            current_loss = running_loss / len(dl.dataset)
            scores = torch.cat(scores)
            y = torch.cat(y)
            yhat = torch.cat(yhat)
            metrics = ut.compute_metrics(y,
                                         yhat,
                                         scores[:, 1])

            if split == "valid":
                self.current_lossva = current_loss
                self.lossva_history.append(self.current_lossva.item())
                self.scoresva = scores
                self.yva = y
                self.yhatva = yhat
                self.metricsva = metrics
            else:
                self.current_losste = current_loss
                self.scoreste = scores
                self.yte = y
                self.yhatte = yhat
                self.metricste = metrics

                # def save_checkpoint(self, epoch):
                #     ckpt_dict = {
                #         'best_ep': epoch,
                #         'state_dict': self.model.state_dict(),
                #         'optim_dict': self.optimizer.state_dict(),
                #         'losstr_history': self.losstr_history,
                #         'lossva_history': self.lossva_history,
                #         'metricsva': self.metricsva,
                #         'metricste': self.metricste}
                #     torch.save(ckpt_dict, self.log_dir / "best.pt")
                #     # training loop

                # def test(self, dl):
                #     best_ckpt = torch.load(log_dir / "best.pt")

                #     model = ut.RumexNet(cfg.model_name)
                #     model.load_state_dict(best_ckpt["state_dict"])
                #     model.to(device)
                #     model.eval()
                #     with torch.no_grad():
                #         score = []
                #         loss = []
                #         y = []
                #         yhat = []
                #         ids = []
                #         for xb, yb, _ in dl:
                #             xb = xb.to(device)
                #             yb = yb.to(device)

                #             # Forward pass
                #             scoreb = model(xb)  # logits
                #             _, yhatb = torch.max(scoreb, 1)
                #             lossb = loss_fn(scoreb, yb)

                #             # book keeping at batch level
                #             score.append(scoreb)
                #             yhat.append(yhatb)
                #             loss.append(lossb)
                #             y.append(yb)

                #         score = torch.cat(score)
                #         loss = torch.cat(loss)
                #         y = torch.cat(y)
                #         yhat = torch.cat(yhat)

                #         predictions = {
                #             'y': y,
                #             'yhat': yhat,
                #             'score': score[:, 1],
                #             'loss': loss
                #         }

                #         # metrics
                #         metrics = ut.compute_metrics(y, yhat, score[:, 1])
                #         metrics['loss'] = torch.mean(loss)

                #     logging.info('test preforance')
                #     logging.info(f"losste:{metrics['loss']:.5f}|" +
                #                  f"acc:{metrics['acc']:.5f}|" +
                #                  f"auc:{metrics['auc']:.5f}|" +
                #                  f"re:{metrics['pre']:.5f}|" +
                #                  f"pre:{metrics['recall']:.5f}|" +
                #                  f"f1:{metrics['f1']:.5f}")
                #     # save test performance
                #     test_ckpt_dict = {'predictions': predictions,
                #                       'metrics': metrics,
                #                       }
                #     torch.save(test_ckpt_dict, log_dir / "test_preds.pt")
                # parser = argparse.ArgumentParser()
                # parser.add_argument('--train_dir', type=str, default="data/train_wa/")
                # parser.add_argument('--valid_dir', type=str, default="data/valid/")
                # parser.add_argument('--test_dir', type=str, default="data/test/")
                # parser.add_argument('--model_name', type=str)
                # parser.add_argument('--max_ep', type=int, default=50)
                # parser.add_argument('--bs', type=int, default=64)
                # parser.add_argument('--gpu', type=bool, default=True)

                # if __name__ == '__main__':
                #     args, unknown = parser.parse_known_args()
                #     trainer(args)
