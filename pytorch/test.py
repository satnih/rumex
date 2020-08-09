# test
import numpy as np
import matplotlib.pyplot as plt

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

best_model_path = '/u/21/hiremas1/unix/postdoc/rumex/src/pytorch/lightning_logs/version_34/checkpoints/epoch=12.ckpt'
trained_model = MyModel.load_from_checkpoint(best_model_path)

from skimage import io as skio
im = skio.imread(
    '/u/21/hiremas1/unix/postdoc/rumex/data_orig/10m/WENR_ortho_Rumex_10m_3_ne.png'
)
im = im[:, :, :3]
ps = 224
nrows = im.shape[0] // ps
ncols = im.shape[1] // ps
im = im[:(nrows * ps), :(ncols * ps), :]
im_pred = np.zeros((im.shape[0], im.shape[1]))
transorm = T.Normalize(imagenet_mean, imagenet_std)
for i in np.arange(nrows):
    print(f'{i}/{nrows}')
    i_start = i * ps
    i_end = (i + 1) * ps
    for j in np.arange(ncols):
        j_start = j * ps
        j_end = (j + 1) * ps

        patch = im[i_start:i_end, j_start:j_end, :]
        patch = patch.transpose((2, 0, 1))
        patch = torch.Tensor(patch)
        # patch = transorm(patch)
        patch = patch.view(1, 3, ps, ps)

        pred = trained_model(patch)
        _, yhat = torch.max(pred, 1)

        im_pred[i_start:i_end, j_start:j_end] = yhat.numpy()

plt.imshow(im)
plt.imshow(im_pred, alpha=0.5)
# testset = ImageFolder(data_path_te, transforms)
# test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size)
# result = trainer.test(test_dataloaders=test_loader)
# print(result)
