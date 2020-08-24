import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json
from lib.utils.vsd import inout, renderer, misc
from lib.config import cfg
import yaml
import cv2


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_mask(data_root, file_name):
    if not os.path.isdir(data_root):
        os.mkdir(data_root)

    mask_dir = os.path.join(data_root, 'mask')
    rgb_dir = os.path.join(data_root, 'rgb')

    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    if not os.path.isdir(rgb_dir):
        os.mkdir(rgb_dir)

    test_data_rgb_ = os.path.join("data/tless/test_primesense","{:02d}","rgb")
    test_data_mask_ = os.path.join("data/tless/test_primesense","{:02d}","mask")

    test_data_rgb = test_data_rgb_.format(file_name)
    test_data_mask = test_data_mask_.format(file_name)

    inds = range(len(os.listdir(test_data_mask)))

    for ind in tqdm.tqdm(inds):
        rgb_path = os.path.join(test_data_rgb, '{:04d}.png'.format(ind))
        rgb = cv2.imread(rgb_path)

        mask_path = os.path.join(test_data_mask, '{:04d}.png'.format(ind))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # change to this number for image maker
        target_mask = np.where(mask == 35, 255, 0)
        cv2.imwrite(os.path.join(mask_dir, "{:d}.png".format(ind)), target_mask)
        cv2.imwrite(os.path.join(rgb_dir, "{:d}.png".format(ind)), rgb)

def record_ann(model_meta, img_id, ann_id, images, annotations, tless_cls, file_name):
    test_data_path_ = os.path.join("data/tless/test_primesense", "{:02d}")
    test_data_path = test_data_path_.format(file_name)

    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']

    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    inds = range(len(os.listdir(rgb_dir)))

    k_info = yaml.load(open(os.path.join(test_data_path, "info.yml"), 'r'), Loader=yaml.FullLoader)
    gt = yaml.load(open(os.path.join(test_data_path, 'gt.yml'), 'r'))

    for ind in tqdm.tqdm(inds):
        K = np.array(k_info[ind]['cam_K']).reshape((3, 3))

        dic_gt = gt[ind]
        for i in range(len(dic_gt)):
            if dic_gt[i]['obj_id'] == tless_cls:
                R = dic_gt[i]['cam_R_m2c']
                t = dic_gt[i]['cam_t_m2c']

        R = np.array(R).reshape(3, 3)
        t = np.array(t) * 0.001
        pose = np.concatenate([R, t[:, None]], axis=1)

        rgb_path = os.path.join(rgb_dir, '{:d}.png'.format(ind))

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
        fps_2d = base_utils.project(fps_3d, K, pose)

        mask_path = os.path.join(mask_dir, '{:d}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': 'cat'})
        annotations.append(anno)

    return img_id, ann_id

def custom_test_to_coco(data_root, tless_cls, file_name):

    if not os.path.isdir(data_root):
        os.mkdir(data_root)

    model_dir = 'data/tless/models_cad'
    obj_path = os.path.join(model_dir, 'obj_{:02d}.ply'.format(tless_cls))
    model_ = inout.load_ply(obj_path)
    model = model_['pts'] / 1000.

    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2

    fps_3d = np.loadtxt(os.path.join('data/tless', 'farthest', 'farthest_{:02}.txt'.format(tless_cls)))

    model_meta = {
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations, tless_cls, file_name)
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'cat'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)