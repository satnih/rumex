import xml.etree.ElementTree as et
import xmltodict
from PIL import Image
from skimage import io as skio
from skimage import exposure
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
# %%
set_type = 'val'
save = 1
src_folder = '/u/21/hiremas1/unix/postdoc/rumex/data_orig/'
dst_folder = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/' + set_type + '/'
if set_type == 'train':
    base_filenames = ['WENR_ortho_Rumex_10m_1_nw', 'WENR_ortho_Rumex_10m_2_sw']
elif set_type == 'val':
    base_filenames = ['WENR_ortho_Rumex_10m_4_se']
elif set_type == 'test':
    base_filenames = ['WENR_ortho_Rumex_10m_3_ne']

rumex_count = 0
other_count = 0
wanted_patch_size = 226
r = wanted_patch_size
c = wanted_patch_size

for base_filename in base_filenames:
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

    objects = xmldict['annotation']['object']
    nobjects = len(objects)

    rumex = []
    for i in range(nobjects):
        obj_i = objects[i]
        if obj_i['name'] == 'rumex':
            temp = obj_i['bndbox']
            bbox = [temp['xmin'], temp['ymin'], temp['xmax'], temp['ymax']]
            bbox = [int(x) for x in bbox]
            rumex.append(bbox)

    im = skio.imread(src_folder + imfile)
    # %% extract 'rumex' patches
    for k, box in enumerate(rumex):
        box = [int(x) for x in box]
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        xmid = xmin + int((xmax - xmin) / 2)
        ymid = ymin + int((ymax - ymin) / 2)
        xmin_new = xmid - int(wanted_patch_size / 2)
        ymin_new = ymid - int(wanted_patch_size / 2)
        xmax_new = xmid + int(wanted_patch_size / 2)
        ymax_new = ymid + int(wanted_patch_size / 2)

        if ((xmin_new > 0) & (ymin_new > 0) & (xmax_new <= im_width) &
            (ymax_new <= im_height)):
            # 1. first extract rumex patch
            # 2. zero-out rumex patch in orthomosaic to extract 'other' patches
            #    later.
            patch = im[ymin_new:ymax_new, xmin_new:xmax_new, :3]
            im[ymin_new:ymax_new, xmin_new:xmax_new, -1] = 0
            if save:
                patch_name = str(rumex_count) + '.tiff'
                skio.imsave(dst_folder + 'rumex/' + patch_name, patch)
            rumex_count += 1

    # %% extract other patches
    im_patches = extract_patches_2d(im, (r, c),
                                    max_patches=300,
                                    random_state=99)

    # keep patches only within the orthomosaic and not overlaping rumex patches
    # NOTE: manual deletion of some extracted patches is required as all rumex
    #       are not labelled in orthomosaic
    valid_patches = []
    for patch in im_patches:
        mask = patch[:, :, -1]
        if np.prod(mask) != 0:
            patch_name = str(other_count) + '.tiff'
            if save:
                skio.imsave(dst_folder + 'other/' + patch_name,
                            patch[:, :, :3])
            other_count += 1
print(f'finished extracting {rumex_count} rumex and {other_count} others')

# %%
