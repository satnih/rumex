import os
import xml.etree.ElementTree as et
import xmltodict
from PIL import Image
from skimage import io as skio
from skimage import exposure
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter
from skimage.util.shape import view_as_blocks
import numpy as np
import pandas as pd
# %%
save = 1
fh = "10m"  # flying height
ps = 256
src_folder = '/u/21/hiremas1/unix/postdoc/rumex/data_orig/' + fh + '/'
dst_folder = 'data_patches_temp/' + fh + '/' + str(ps) + '/'

base_filename = 'WENR_ortho_Rumex_10m_4_se'


# 10m files
# base_filename = 'WENR_ortho_Rumex_10m_1_nw'
# base_filename = 'WENR_ortho_Rumex_10m_2_sw'
# base_filename = 'WENR_ortho_Rumex_10m_3_ne'
# base_filename = 'WENR_ortho_Rumex_10m_4_se'

# 15m files
# base_filename = 'WENR_ortho_Rumex_15m_2_sw'
# base_filename = 'WENR_ortho_Rumex_15m_4_se'

base_filenames = {"10m": ['WENR_ortho_Rumex_10m_1_nw',
                          'WENR_ortho_Rumex_10m_2_sw',
                          'WENR_ortho_Rumex_10m_3_ne',
                          'WENR_ortho_Rumex_10m_4_se'],
                  "15m": ['WENR_ortho_Rumex_15m_2_sw',
                          'WENR_ortho_Rumex_15m_4_se']}

for base_filename in base_filenames[fh]:
    quadrant = base_filename.split("_")[-1]

    rumex_count = 0
    other_count = 0

    print(base_filename)
    imfile = base_filename + '.png'
    xmlfile = base_filename + '.xml'
    root = et.parse(src_folder + xmlfile).getroot()
    xmlstr = et.tostring(root, encoding='utf-8', method='xml')
    xmldict = dict(xmltodict.parse(xmlstr))

    im_size = xmldict['annotation']['size']
    im_height = int(im_size['height'])
    im_width = int(im_size['width'])
    im_depth = int(im_size['depth'])

    # load image and resize it to be multiple of ps
    npatches_w = im_width // ps
    npatches_h = im_height // ps
    im = skio.imread(src_folder + imfile)

    im = im[:npatches_h * ps, :npatches_w * ps, :]
    im_patches = np.squeeze(view_as_blocks(im, (ps, ps, 4)))

    im_label = np.zeros(im.shape[:2])
    label_patches = view_as_blocks(im_label, (ps, ps))

    # get annotations form xml file
    objects = xmldict['annotation']['object']
    nobjects = len(objects)
    brumex = []
    for i in range(nobjects):
        obj_i = objects[i]
        if obj_i['name'] == 'rumex':
            temp = obj_i['bndbox']
            bbox = [temp['xmin'], temp['ymin'], temp['xmax'], temp['ymax']]
            bbox = [int(x) for x in bbox]

            xmin_r = bbox[0] // ps
            ymin_r = bbox[1] // ps
            xmax_r = bbox[2] // ps
            ymax_r = bbox[3] // ps

            if (xmax_r < npatches_w) & (ymax_r < npatches_h):
                if xmax_r - xmin_r >= 1:
                    x_patches_with_rumex = list(np.arange(xmin_r, xmax_r + 1))
                else:
                    x_patches_with_rumex = [xmin_r]

                if ymax_r - ymin_r >= 1:
                    y_patches_with_rumex = list(np.arange(ymin_r, ymax_r + 1))
                else:
                    y_patches_with_rumex = [ymin_r]

                for col in x_patches_with_rumex:
                    for row in y_patches_with_rumex:
                        im_label[row:(row + ps), col:(col + ps)] = 1
                        label_patches[row, col] = 1

    for row in np.arange(npatches_h):
        for col in np.arange(npatches_w):
            # create patch id and name for storing
            patch_id = npatches_w * row + col
            if patch_id < 10:
                patch_id_str = "000"+str(patch_id)
            elif patch_id >= 10 and patch_id < 100:
                patch_id_str = "00"+str(patch_id)
            elif patch_id >= 100 and patch_id < 1000:
                patch_id_str = "0"+str(patch_id)
            patch_name = patch_id_str + "_" + quadrant + '_patch.png'

            patch = im_patches[row, col]
            patch_mask = patch[:, :, -1]
            label = label_patches[row, col]
            if np.sum(patch_mask == 255) == np.prod(patch_mask.shape):
                if np.sum(label) == ps * ps:
                    dst = dst_folder + base_filename + '/rumex/'
                else:
                    dst = dst_folder + base_filename + '/other/'
            else:  # outside field
                dst = dst_folder + base_filename + '/outside/'

            if not os.path.exists(dst):
                os.makedirs(dst)
            if save:
                skio.imsave(dst + patch_name, patch[:, :, :3])
                # print(dst + patch_name)


# fig, ax = plt.subplots(1)
# ax.imshow(im)
# ax.imshow(im_label, alpha=0.5)
# # fig.savefig('test.png')
