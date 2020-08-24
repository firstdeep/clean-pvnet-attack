import os
import numpy as np
import tqdm
from lib.utils.vsd import inout
from pytless import renderer
import yaml
import cv2

def get_mask(data_root, tless_cls):

    model_dir = 'data/tless/models_cad'
    obj_path = os.path.join(model_dir, 'obj_{:02d}.ply'.format(tless_cls))
    model = inout.load_ply(obj_path)
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    inds = range(len(os.listdir(rgb_dir)))

    info = yaml.load(open(os.path.join(data_root, "info.yml"), 'r'), Loader=yaml.FullLoader)
    gt = yaml.load(open(os.path.join(data_root, 'gt.yml'), 'r'), Loader=yaml.FullLoader)

    for ind in tqdm.tqdm(inds):
        rgb_path = os.path.join(rgb_dir, '{:04d}.png'.format(ind))
        rgb = cv2.imread(rgb_path)
        im_size = (rgb.shape[1], rgb.shape[0])

        K = np.array(info[ind]['cam_K']).reshape((3, 3))
        R = np.array(gt[ind][0]['cam_R_m2c']).reshape(3, 3)
        t = np.array(gt[ind][0]['cam_t_m2c'])

        surf_color = (255, 0, 0)
        ren_rgb = renderer.render(model, im_size, K, R, t,
                                  surf_color=surf_color, mode='rgb')

        gray = cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float)
        vis_gray = np.where(gray > 0, 255, 0)
        cv2.imwrite(mask_dir + "/{:d}.png".format(ind), vis_gray.astype(np.uint8))