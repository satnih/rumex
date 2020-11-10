# %%
# # %% Test model------------------------------------------------------------
import torch
import logging
import itertools
import numpy as np
import utils as ut
import pandas as pd
from time import time
from torch import optim
from pathlib import Path
from skimage import io as skio
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import accuracy_score, f1_score, precision_score
device = torch.device("cpu")

model_name = "resnet"
model_dir = "logs/triton/"+model_name+"/"
data_dir = "~/postdoc/rumex/data_patches/15m/256/WENR_ortho_Rumex_15m_2_sw/"
result_dir = "results/15m/"

ckpt_dict = torch.load(model_dir+"best.pt")
model = ut.RumexNet(model_name)
model.load_state_dict(ckpt_dict["state_dict"])
model = model.to(device)

# test dataset and data loader
dste = ut.RumexDataset(data_dir+'train/', train_flag=False)
dlte = ut.test_loader(dste, 64)
# %% make predictions
nte = len(dste)
n1te = dste.rumex.targets.count(1)
n0te = dste.rumex.targets.count(0)

yte = []
score1te = []
score0te = []
losste = []
yhatte = []
fnamete = []
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

for xteb, yteb, idb, fnameb in dlte:
    xteb = xteb.to(device)
    yteb = yteb.to(device)

    # Forward pass
    score_te_b = model(xteb)  # logits
    loss_te_b = loss_fn(score_te_b, yteb)
    _, yhat_te_b = torch.max(score_te_b, 1)

    # book keeping at batch level
    yte.append(yteb.detach().numpy().tolist())

    score0te.append(score_te_b.detach().numpy()[:, 0].tolist())
    score1te.append(score_te_b.detach().numpy()[:, 1].tolist())

    losste.append(loss_te_b.detach().numpy().tolist())
    yhatte.append(yhat_te_b.detach().numpy().tolist())
    fnamete.append(list(fnameb))

yte = list(pd.core.common.flatten(yte))
score0te = list(pd.core.common.flatten(score0te))
score1te = list(pd.core.common.flatten(score1te))
losste = list(pd.core.common.flatten(losste))
yhatte = list(pd.core.common.flatten(yhatte))
fnamete = list(pd.core.common.flatten(fnamete))

pred_test = {"fname": fnamete,
             "yte": yte,
             "yhatte": yhatte,
             "scorete0": score0te,
             "scorete1": score1te,
             "losste": losste}

df_test = pd.DataFrame.from_dict(pred_test)
df_test.to_csv(result_dir + model_name+"_test_predictions.csv")

# %%
temp = ckpt_dict['predictions']
pred_valid = {}
pred_valid["yva"] = temp['yva'].detach().cpu().numpy()
pred_valid["yhatva"] = temp['yhatva'].detach().cpu().numpy()
pred_valid["score1va"] = temp['scoreva'].detach().cpu().numpy()
pred_valid["lossva"] = temp['lossva'].detach().cpu().numpy()
df_valid = pd.DataFrame.from_dict(pred_valid)
df_valid.to_csv(result_dir + model_name+"_valid_predictions.csv")


# df.sort_values("loss", ascending=False, ignore_index=True, inplace=True)


# def display_image_grid(im_path_list, titles):
#     '''
#     Given list of image paths, extract its RGB bands, rescale the values for display
#         Input: 1) im_path_list: str containg full path to the image file.
#     '''
#     nimages = len(im_path_list)
#     nrows = 10
#     ncols = 5
#     fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 30))
#     for i in range(nimages):
#         # load image
#         im = skio.imread(im_path_list[i])
#         titlei = titles[i]
#         # rescale intensities
#         # im = im[3:0:-1].transpose(2, 1, 0)
#         im = rescale_intensity(im)

#         # plot image
#         row = i // ncols
#         col = i % ncols
#         axarr[row, col].axis("off")
#         axarr[row, col].imshow(im, aspect="auto")
#         axarr[row, col].set_title(titlei)
#     plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0,
#                         right=1, bottom=0, top=0.98)
#     plt.show()
#     fig.savefig('top_loss_1.png')
#     return


# display_image_grid(file_paths, titles)

# N = 50
# file_paths = []
# titles = []
# for i in np.arange(N):
#     fi = df.iloc[i, 0]
#     folder = fi.split("_")[0]
#     fpi = os.path.join(data_dir, "test_uncleaned", folder, fi)
#     titlei = f"true:{df.iloc[i, 1]}|pred:{df.iloc[i, 2]}|loss:{df.iloc[i, 5]:.3f}"
#     file_paths.append(fpi)
#     titles.append(titlei)


# acc_te = accuracy_score(yte, yhatte)
# f1_te = f1_score(yte, yhatte)
# pre_te = precision_score(yte, yhatte)
# recall_te = recall_score(yte, yhatte)
# auc_te = roc_auc_score(yte, scorete[:, 1])


# dict_metrics = {"dummy_acc": nother/ntest,
#                 "acc": acc_te,
#                 "f1": f1_te,
#                 "pre": pre_te,
#                 "rec": recall_te,
#                 "auc": auc_te}

# print(f"dummy_acc:{nother/ntest}, acc:{acc_te:.5f}, f1:{f1_te:.5f}")

# # print(f"acc:{acc_te:.5f}|auc:{auc_te:.5f}|f1:{f1_te:.5f}" +
# #       f"|pre:{pre_te:.5f}|recall:{recall_te:.5f}")

# # %%

# %%
