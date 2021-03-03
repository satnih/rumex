import os
import torch
import utils as ut
import numpy as np
import pandas as pd
from skimage import io as skio
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from torchvision import transforms as T
from skimage.draw import rectangle_perimeter
from skimage.util.shape import view_as_blocks
device = torch.device("cuda")

# %%
save = 0
fh = "15m"  # flying height
ps = 256

model_path = "results/trained_models/"
src_folder = '/u/21/hiremas1/unix/postdoc/rumex/data_orig/' + fh + '/'

# test_filenames = {"10m": ['WENR_ortho_Rumex_10m_1_nw',
#                           'WENR_ortho_Rumex_10m_4_se'],
#                   "15m": ['WENR_ortho_Rumex_15m_2_sw']}

# for test_file in test_filenames[fh]:
test_file = "'WENR_ortho_Rumex_15m_2_sw'"
rumex_count = 0
other_count = 0

# load image and resize it to be multiple of ps
im = skio.imread(src_folder + 'WENR_ortho_Rumex_15m_2_sw.png')

im_height = im.shape[0]
im_width = im.shape[1]
im_depth = im.shape[2]

npatches_w = im_width // ps
npatches_h = im_height // ps

# crop image to multiple of ps
im = im[:npatches_h * ps, :npatches_w * ps, :]
pred = np.zeros(im.shape[:2])
pred_bbox = []

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std)
])

model_name = "mobilenet"
trainer = torch.load(model_path + 'mobilenet_seed_0_256_trainer.pt',
                     map_location=device)
model = ut.RumexNet(model_name)
# %%
model.load_state_dict(trainer.model.state_dict())
model.to(device)
model.eval()
with torch.no_grad():
    for row in np.arange(npatches_h):
        for col in np.arange(npatches_w):
            ymin = row*ps
            ymax = (row + 1)*ps

            xmin = col*ps
            xmax = (col+1)*ps

            patch = im[ymin:ymax, xmin:xmax]
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
                # print(yhat.item(), end=",")
                pred[ymin:ymax, xmin:xmax] = yhat.item()
                if yhat.item() == 1:
                    pred_bbox.append([xmin, ymin, xmax, ymax])
plt.imshow(pred)

# %%
