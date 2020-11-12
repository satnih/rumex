import os
import torch
import numpy as np
import pandas as pd
import utils as ut
from skimage import io as skio
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from torchvision import transforms as T
from skimage.draw import rectangle_perimeter
from skimage.util.shape import view_as_blocks
device = torch.device("cuda")

# %%
save = 0
fh = "10m"  # flying height
ps = 256

model_path = "results/10m/from_triton/"
src_folder = '/u/21/hiremas1/unix/postdoc/rumex/data_orig/' + fh + '/'
dst_folder = 'results/predictions/' + fh + '/' + str(ps) + '/'

test_filenames = {"10m": ['WENR_ortho_Rumex_10m_1_nw',
                          'WENR_ortho_Rumex_10m_4_se'],
                  "15m": ['WENR_ortho_Rumex_15m_2_sw',
                          'WENR_ortho_Rumex_15m_4_se']}

# for test_file in test_filenames[fh]:
test_file = "WENR_ortho_Rumex_10m_1_nw"
rumex_count = 0
other_count = 0

# load image and resize it to be multiple of ps
im = skio.imread(src_folder + 'WENR_ortho_Rumex_10m_1_nw.png')

im_height = im.shape[0]
im_width = im.shape[1]
im_depth = im.shape[2]

npatches_w = im_width // ps
npatches_h = im_height // ps

# crop image to multiple of ps
im = im[:npatches_h * ps, :npatches_w * ps, :]
pred = np.zeros(im.shape[:2])

sz = 224
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(sz),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std)
])

model_name = "resnet"
trainer = torch.load(model_path + model_name+"_trainer.pt")
model = ut.RumexNet(model_name)
# %%
model.load_state_dict(trainer.model.state_dict())
model.to(device)
model.eval()
with torch.no_grad():
    for row in np.arange(npatches_h):
        for col in np.arange(npatches_w):
            row_start = row*ps
            row_end = (row + 1)*ps

            col_start = col*ps
            col_end = (col+1)*ps

            patch = im[row_start:row_end, col_start:col_end]
            patch_rgb = patch[:, :, :3]
            patch_mask = patch[:, :, -1]

            # 255 indicate inside the fild
            # if entire patch is inside the field then
            if np.sum(patch_mask == 255) == np.prod(patch_mask.shape):
                x = tfms(patch_rgb)
                x = x.unsqueeze(dim=0)
                x = x.to(device)
                score = model(x)  # logits
                _, yhat = torch.max(score, 1)
                print(yhat.item(), end=",")
                pred[row_start:row_end, col_start:col_end] = yhat.item()
plt.imshow(pred)

# %%
