# %%
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from lightning import Net
data_dir = "/u/21/hiremas1/unix/postdoc/rumex/data_for_fastai_cleaned/"
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# resnet: lr [1e-4, 1e-3]
# mobilenet: lr [1e-4, 1e-3]
# shufflenet: lr [1e-3, 1e-1]
# mnasnet: lr [5e-3, 7e-2]


def get_test_loader(data_dir):
    # OPTIONAL
    transforms = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])
    testset = ImageFolder(data_dir+"test_uncleaned", transforms)
    dlte = DataLoader(testset,
                      batch_size=32,
                      shuffle=True,
                      num_workers=1)
    return dlte


dlte = get_test_loader(data_dir)
# %%
model_name = "mobilenet"
min_lr = 1e-4
max_lr = 1e-3
best_epoch = 5

ckpt_file = str(best_epoch)+".ckpt"
ckpt_dir = model_name + "_adamw_logs/version_1/checkpoints/epoch="+ckpt_file
ckpt = torch.load(ckpt_dir)
mdl_best = Net(model_name, min_lr=min_lr, max_lr=max_lr)
mdl_best.load_state_dict(ckpt["state_dict"])
mdl_best.eval()
yhat = []
y = []
logits = []
for xb, yb in dlte:
    logitsb = mdl_best(xb)
    _, yhatb = torch.max(logitsb, dim=1)
    y.append(yb)
    yhat.append(yhatb)
    logits.append(logitsb)

y = torch.cat(y).detach().numpy()
yhat = torch.cat(yhat).detach().numpy()
logits = torch.cat(logits).detach().numpy()

dummy_acc = 1-sum(y)/len(y)
acc_te = sum(y == yhat)/len(y)
print(f"dummy: {dummy_acc}, {model_name}: {acc_te}")

# %%
