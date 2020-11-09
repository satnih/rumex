import torch
import logging
import argparse
import numpy as np
import utils as ut
from time import time
from torch import optim
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter

# def train(model, dl, optimizer, scheduler, loss_fn, device):
#     model.to(device)
#     model.train()
#     running_loss = 0.0
#     for xb, yb in dl:
#         # Forward pass
#         xb = xb.to(device)
#         yb = yb.to(device)
#         scoreb = model(xb)
#         lossb = loss_fn(scoreb, yb)

#         # Backward and optimize
#         lossb_mean = torch.mean(lossb)
#         lossb_mean.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # scheduler.step()
#         running_loss += torch.sum(lossb)

#     losstr_mean = running_loss / len(dl.dataset)
#     return losstr_mean


def test(log_dir, cfg, dl, loss_fn, device):
    best_ckpt = torch.load(log_dir / "best.pt")

    model = ut.RumexNet(cfg.model_name)
    model.load_state_dict(best_ckpt["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        score = []
        loss = []
        y = []
        yhat = []
        ids = []
        for xb, yb, _ in dl:
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
            y.append(yb)

        score = torch.cat(score)
        loss = torch.cat(loss)
        y = torch.cat(y)
        yhat = torch.cat(yhat)

        predictions = {
            'y': y,
            'yhat': yhat,
            'score': score[:, 1],
            'loss': loss
        }

        # metrics
        metrics = ut.compute_metrics(y, yhat, score[:, 1])
        metrics['loss'] = torch.mean(loss)

    logging.info('test preforance')
    logging.info(f"losste:{metrics['loss']:.5f}|" +
                 f"acc:{metrics['acc']:.5f}|" +
                 f"auc:{metrics['auc']:.5f}|" +
                 f"re:{metrics['pre']:.5f}|" +
                 f"pre:{metrics['recall']:.5f}|" +
                 f"f1:{metrics['f1']:.5f}")

    # save test performance
    test_ckpt_dict = {'predictions': predictions,
                      'metrics': metrics,
                      }
    torch.save(test_ckpt_dict, log_dir / "test_preds.pt")


def trainer(cfg):
    gpu = cfg.gpu & torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    eq50 = '=' * 50
    exp_name = cfg.model_name
    log_dir = ut.set_logger(Path.cwd() / cfg.log_dir / exp_name)
    # writer = SummaryWriter(log_dir=log_dir)
    logging.info(eq50 + 'training started' + eq50)
    logging.info(cfg)

    # dataset and data loader
    dstr = ut.RumexDataset(cfg.train_dir)
    dsva = ut.RumexDataset(cfg.valid_dir)
    dltr = ut.train_loader(dstr, cfg.bs)
    dlva = ut.test_loader(dsva, 2*cfg.bs)

    ntr = len(dstr)
    n1tr = dstr.rumex.targets.count(1)
    n0tr = dstr.rumex.targets.count(0)

    nva = len(dsva)
    n1va = dsva.rumex.targets.count(1)
    n0va = dsva.rumex.targets.count(0)
    logging.info(f"(train):{ntr,n1tr,n0tr}, (valid):{nva,n1va,n0va}")

    # model and optimizer
    model = ut.RumexNet(cfg.model_name)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # optimized lr ranges for different models
    # resnet: 1e-4, 1e-3
    # mobilenet: 1e-4, 7e-3
    # densenet: 1e-4, 1e-3
    # mnasnet: 1e-3, 1e-2
    # shufflenet: 5e-3, 5e-2

    lr_dict = {"resnet": 1e-3,
               "mobilenet": 7e-3,
               "densenet": 1e-3,
               "mansnet": 1e-2,
               "shufflenet": 5e-2}

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr_dict[cfg.model_name])

    # training loop

    lossva_history = []
    losstr_history = []
    for ep in np.arange(cfg.num_epochs):
        start = time()

        #### train model ##########
        model.train()
        running_losstr = 0.0
        for xtrb, ytrb, _, _ in dltr:
            # Forward pass
            xtrb = xtrb.to(device)
            ytrb = ytrb.to(device)
            scorebtr = model(xtrb)
            # loss of each elem in batch
            losstrb = loss_fn(scorebtr, ytrb)

            # Backward and optimize
            lossbtr_mean = torch.mean(losstrb)
            lossbtr_mean.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            running_losstr += torch.sum(losstrb)

        losstr_mean = running_losstr / ntr

        ##### validate model ##########
        model.eval()
        with torch.no_grad():
            fnameva = []
            scoreva = []
            yva = []
            yhatva = []
            lossva = []
            for xvab, yvab, _, fnamevab in dlva:
                xvab = xvab.to(device)
                yvab = yvab.to(device)

                # Forward pass
                scorevab = model(xvab)  # logits
                lossvab = loss_fn(scorevab, yvab)
                _, yhatvab = torch.max(scorevab, 1)

                # book keeping at batch level
                fnameva.append(fnamevab)
                lossva.append(lossvab)
                scoreva.append(scorevab)
                yhatva.append(yhatvab)
                yva.append(yvab)

            lossva = torch.cat(lossva)
            lossva_mean = torch.mean(lossva)  # mean validation loss for ep

            scoreva = torch.cat(scoreva)
            yva = torch.cat(yva)
            yhatva = torch.cat(yhatva)
            fname = [b for a in fnameva for b in a]  # flatten list of tuples

            predictions = {
                'yva': yva,
                'yhatva': yhatva,
                'scoreva': scoreva[:, 1],
                'lossva': lossva
            }

            # metrics
            metrics = ut.compute_metrics(yva, yhatva, scoreva[:, 1])

            ##### early stopping and checkpoint saving ##########
            # from deep larning book
            if ep == 0:
                es_ep = ep  # i in  book
                patience_counter = 0  # j in  book
                best_mean_lossva = np.inf  # nu in book
            elif ep > 10:
                if lossva_mean < best_mean_lossva:
                    # update early stopping variables
                    best_mean_lossva = lossva_mean
                    patience_counter = 0  # reset patience counter
                    es_ep = ep  # early stopping epoch (i* in book)

                    # save best model upto now
                    ckpt_dict = {
                        'ep': es_ep,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'predictions': predictions,
                        'metrics': metrics,
                        'best_mean_lossva': lossva_mean,
                        'losstr_history': torch.stack(losstr_history),
                        'lossva_history': torch.stack(lossva_history)
                    }
                    torch.save(ckpt_dict, log_dir / "best.pt")
                else:
                    patience_counter += 1  # increament patience counter

            # # tensorboad logging
            # writer.add_scalar('train/loss', loss, ep)
            # for key in metrics.keys():
            #     name = 'val/' + key
            #     writer.add_scalar(name, metrics[key], ep)

            et = time() - start
            logging.info(f"ep:{ep}|" +
                         f"et:{et:.0f}|" +
                         f"patience_count:{patience_counter}|" +
                         f"losstr:{losstr_mean:.5f}|" +
                         f"lossva:{lossva_mean:.5f}|" +
                         f"acc:{metrics['acc']:.5f}|" +
                         f"f1:{metrics['f1']:.5f}")

            # history
            losstr_history.append(losstr_mean)
            lossva_history.append(lossva_mean)

            # loss_history[ep, 0] = losstr_mean  # training loss
            # loss_history[ep, 1] = lossva_mean  # validation loss

            if patience_counter >= args.patience:
                logging.info(
                    f"***stopping early at {ep}/{cfg.num_epochs}****")
                break

    # torch.save(loss_history, log_dir / "loss_history.pt")
    logging.info(eq50 + 'training ended' + eq50)


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="data/10m/256/train_wa/")
parser.add_argument('--valid_dir', type=str, default="data/10m/256/valid/")
parser.add_argument('--test_dir', type=str, default="data/10m/256/test/")
parser.add_argument('--log_dir', type=str, default="logs_temp")
parser.add_argument('--model_name', type=str, default="shufflenet")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--gpu', type=bool, default=True)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    trainer(args)
