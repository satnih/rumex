# code modified from: https://captum.ai/tutorials/CIFAR_TorchVision_Interpret
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import utils as ut
import torchvision
import matplotlib.pyplot as plt
from rumex_dataset import RumexDataset
from rumex_model import RumexNet
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import GradientShap
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(img, transpose=True):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def subset_data(x, y, idx, fname, n):
    return x[:n], y[:n], idx[:n], fname[:n]


def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                              )

    return tensor_attributions


model_name = 'shufflenet_v2'
cfg = ut.load_config('config.yaml')
log_dir = 'logs_from_triton/' + model_name + '_logs/'
data_dir = cfg['data_dir']+'valid/'

# load best model
model_file = log_dir + 'best.pt'
model = torch.load(model_file)
model = model.cpu()

# load validation data
dsva = RumexDataset(data_dir, train_flag=False)
dlva = dsva.make_data_loader()
dataiter = iter(dlva)

x, y, idx, fname = dataiter.next()
classes = ['o', 'r']
x1, y1, idx1, fname1 = subset_data(x, y, idx, fname, 8)

nimages = len(x1)
output = model(x1)
output = F.softmax(output, dim=1)
score, yhat = torch.topk(output, 1)


imshow(torchvision.utils.make_grid(x1))
print('true: ', ' '.join('%5s' % classes[y1[j]] for j in range(nimages)))
print('pred: ', ' '.join('%5s' % classes[yhat[j]] for j in range(nimages)))
print('score: ', ' '.join('%.3f' % score[j].item() for j in range(nimages)))


ind = 3
input_eg = x1[ind].unsqueeze(0)
target_eg = yhat[ind].unsqueeze(0)

# integrated gradients
integrated_gradients = IntegratedGradients(model)
attrib_ig = integrated_gradients.attribute(
    input_eg, target=target_eg, n_steps=200)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

_ = viz.visualize_image_attr(np.transpose(attrib_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                             np.transpose(
                                 input_eg.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)
viz.visualize_image_attr()


torch.manual_seed(0)
np.random.seed(0)

gradient_shap = GradientShap(model)

# Defining baseline distribution of images
rand_img_dist = torch.cat([input * 0, input * 1])

attributions_gs = gradient_shap.attribute(input_eg,
                                          n_samples=50,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=target_eg)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      np.transpose(
                                          input_eg.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)
