# from rumex_model import RumexNet
# # # %% Test model------------------------------------------------------------
# model_name = "shufflenet"
# data_dir = ""
# ckpt_dir = "logs_from_triton/shufflenet_logs/best.pt"
# model = RumexNet(model_name)
# model.load_state_dict(torch.load(ckpt_dir)
# model.eval()

# dlte=RumexDataset(data_dir+'test_uncleanded/', tfms_te).make_data_loader(bs)
# ntest=len(dlte.dataset)
# nrumex=np.sum(dlte.dataset.targets)
# nother=ntest - nrumex

# y_te=[]
# score_te=[]
# loss_te=[]
# yhat_te=[]
# for ib, (xteb, yteb) in enumerate(dlte):
#     xteb=xteb.to(device)
#     yteb=yteb.to(device)

#     # Forward pass
#     score_te_b=model(xteb)  # logits
#     loss_tr_b=loss_fn(score_te_b, yteb)
#     _, yhat_te_b=torch.max(score_te_b, 1)

#     # book keeping at batch level
#     y_te.append(yteb)
#     score_te.append(score_te_b)
#     yhat_te.append(yhat_te_b)

# # predictions and  metrics
# y_te=torch.cat(y_te).cpu().detach()
# score_te=torch.cat(score_te).cpu().detach()
# loss_te=torch.cat(loss_te).cpu().detach()
# yhat_te=torch.cat(yhat_te).cpu().detach()

# acc_te=accuracy_score(y_te, yhat_te)
# f1_te=f1_score(y_te, yhat_te)
# pre_te=precision_score(y_te, yhat_te)
# recall_te=recall_score(y_te, yhat_te)
# auc_te=roc_auc_score(y_te, score_te[:, 1])

# print(f"{model_name}|#rumex:{nrumex}|#other:{nother}")
# print(f"acc:{acc_te:.5f}|auc:{auc_te:.5f}|f1:{f1_te:.5f}" +
#       f"|pre:{pre_te:.5f}|recall:{recall_te:.5f}")

# %%


# # %% Test model------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
from rumex_dataset import RumexDataset
from rumex_model import RumexNet
from models import load_pretrained
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score, recall_score
from skimage import io as skio
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
device = torch.device("cpu")

data_dir = "/u/21/hiremas1/unix/postdoc/rumex/data_for_fastai_cleaned/"
dste = RumexDataset(data_dir+'test_uncleaned/', train_flag=0)
dlte = dste.make_data_loader(bs=32)
ntest = len(dste)
nrumex = np.sum(dste.rumex.targets)
nother = ntest - nrumex
model = torch.load("logs_from_triton/shufflenet_logs/best.pt")
model.to(device)

y_te = []
score_te = []
loss_te = []
yhat_te = []
fname_te = []
loss_fn = nn.CrossEntropyLoss(reduction="none")
for xteb, yteb, idb, fnameb in dlte:
    xteb = xteb.to(device)
    yteb = yteb.to(device)

    # Forward pass
    score_te_b = model(xteb)  # logits
    loss_te_b = loss_fn(score_te_b, yteb)
    _, yhat_te_b = torch.max(score_te_b, 1)

    # book keeping at batch level
    y_te.append(yteb)
    score_te.append(score_te_b)
    loss_te.append(loss_te_b)
    yhat_te.append(yhat_te_b)
    fname_te.append(list(fnameb))

# predictions and  metrics
y_te = torch.cat(y_te).detach().numpy()
score_te = torch.cat(score_te).detach().numpy()
loss_te = torch.cat(loss_te).detach().numpy()
yhat_te = torch.cat(yhat_te).detach().numpy()
fname_te = list(itertools.chain(*fname_te))

pred = {"fname": fname_te,
        "y": y_te,
        "yhat": yhat_te,
        "score0": score_te[:, 0],
        "score1": score_te[:, 1],
        "loss": loss_te}

df = pd.DataFrame.from_dict(pred)
df.sort_values("loss", ascending=False, ignore_index=True, inplace=True)


def display_image_grid(im_path_list, titles):
    '''
    Given list of image paths, extract its RGB bands, rescale the values for display
        Input: 1) im_path_list: str containg full path to the image file.
    '''
    nimages = len(im_path_list)
    nrows = 10
    ncols = 5
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 30))
    for i in range(nimages):
        # load image
        im = skio.imread(im_path_list[i])
        titlei = titles[i]
        # rescale intensities
        # im = im[3:0:-1].transpose(2, 1, 0)
        im = rescale_intensity(im)

        # plot image
        row = i // ncols
        col = i % ncols
        axarr[row, col].axis("off")
        axarr[row, col].imshow(im, aspect="auto")
        axarr[row, col].set_title(titlei)
    plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0,
                        right=1, bottom=0, top=0.98)
    plt.show()
    fig.savefig('top_loss_1.png')
    return


display_image_grid(file_paths, titles)

N = 50
file_paths = []
titles = []
for i in np.arange(N):
    fi = df.iloc[i, 0]
    folder = fi.split("_")[0]
    fpi = os.path.join(data_dir, "test_uncleaned", folder, fi)
    titlei = f"true:{df.iloc[i, 1]}|pred:{df.iloc[i, 2]}|loss:{df.iloc[i, 5]:.3f}"
    file_paths.append(fpi)
    titles.append(titlei)


acc_te = accuracy_score(y_te, yhat_te)
f1_te = f1_score(y_te, yhat_te)
pre_te = precision_score(y_te, yhat_te)
recall_te = recall_score(y_te, yhat_te)
auc_te = roc_auc_score(y_te, score_te[:, 1])


dict_metrics = {"dummy_acc": nother/ntest,
                "acc": acc_te,
                "f1": f1_te,
                "pre": pre_te,
                "rec": recall_te,
                "auc": auc_te}

print(f"dummy_acc:{nother/ntest}, acc:{acc_te:.5f}, f1:{f1_te:.5f}")

# print(f"acc:{acc_te:.5f}|auc:{auc_te:.5f}|f1:{f1_te:.5f}" +
#       f"|pre:{pre_te:.5f}|recall:{recall_te:.5f}")

# %%
