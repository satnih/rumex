import torch
import logging
import argparse
import numpy as np
import utils as ut
from time import time
from torch import optim
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter


def trainer(cfg):
    gpu = cfg.gpu & torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    eq50 = '=' * 50
    exp_name = cfg.model_name
    log_dir = ut.set_logger(Path.cwd() / "logs" / exp_name)
    # writer = SummaryWriter(log_dir=log_dir)
    logging.info(eq50 + 'training started' + eq50)
    logging.info(cfg)

    # dataset and data loader
    dstr = ut.RumexDataset(cfg.data_dir+'valid/', train_flag=True)
    dsva = ut.RumexDataset(cfg.data_dir+'train/', train_flag=False)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # training loop
    best_mean_lossva = np.inf
    loss_history = torch.zeros((cfg.num_epochs, 2))
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

            # flatten
            predictions = {
                'yva': yva,
                'yhatva': yhatva,
                'scoreva': scoreva[:, 1],
                'lossva': lossva
            }

            # metrics
            metrics = ut.compute_metrics(yva, yhatva, scoreva[:, 1])

            ##### checkpoint saving ##########
            if lossva_mean < best_mean_lossva:
                best_mean_lossva = lossva_mean
                ckpt_dict = {
                    'ep': ep,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'predictions': predictions,
                    'metrics': metrics,
                    'best_mean_lossva': lossva_mean
                }
                torch.save(ckpt_dict, log_dir / "best.pt")

            # # tensorboad logging
            # writer.add_scalar('train/loss', loss, ep)
            # for key in metrics.keys():
            #     name = 'val/' + key
            #     writer.add_scalar(name, metrics[key], ep)

            et = time() - start
            logging.info(f"ep:{ep}|" +
                         f"et:{et:.0f}|" +
                         f"losstr:{losstr_mean:.5f}|" +
                         f"lossva:{lossva_mean:.5f}|" +
                         f"acc:{metrics['acc']:.5f}|" +
                         f"re:{metrics['pre']:.5f}|" +
                         f"pre:{metrics['recall']:.5f}|" +
                         f"f1:{metrics['f1']:.5f}")
            # store history
            loss_history[ep, 0] = losstr_mean  # training loss
            loss_history[ep, 1] = lossva_mean  # validation loss

    torch.save(loss_history, log_dir / "loss_history.pt")
    logging.info(eq50 + 'training ended' + eq50)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default="~/postdoc/rumex/data256_for_training/")
parser.add_argument('--model_name', type=str, default="shufflenet")
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--gpu', type=bool, default=True)

# optimal LR ranges for different models: use max_lr if not using scheduler
# resnet: [1e-4, 1e-3]
# densenet: [1e-4, 1e-3]
# mnasnet: [1e-4, 1e-3]
# mobilenet: [1e-4, 7e-3]
# shufflenet: [5e-3, 5e-2]

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    trainer(args)
