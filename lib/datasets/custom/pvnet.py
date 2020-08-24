import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
from lib.config import cfg
from lib.utils.tless import tless_train_utils, tless_utils, tless_config, tless_pvnet_utils
import glob
import cv2
import imgaug
from lib.utils import data_utils
import time
import glob

_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std

def cut_and_paste(img, mask, train_img):
    ys, xs = np.nonzero(mask)
    # y_min, y_max = np.min(ys), np.max(ys)
    # x_min, x_max = np.min(xs), np.max(xs)
    # h, w = y_max - y_min, x_max - x_min
    # img_h, img_w = img.shape[0], img.shape[1]

    # dst_y, dst_x = np.random.randint(0, img_h-h), np.random.randint(0, img_w-w)
    # dst_ys, dst_xs = ys - y_min + dst_y, xs - x_min + dst_x
    #
    # train_img[dst_ys, dst_xs] = img[ys, xs]
    # train_mask[dst_ys, dst_xs] = mask_id
    train_img[ys, xs] = img[ys, xs]
    return train_img, mask

class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

        self.bg_paths = np.array(glob.glob('data/sun/JPEGImages/*.jpg'))

        self.other_obj_coco = COCO('data/tless/train_primesense/assets/train.json')
        cat_ids = self.other_obj_coco.getCatIds()
        cat_ids.remove(int(cfg.tless_cls))
        self.oann_ids = self.other_obj_coco.getAnnIds(catIds=cat_ids)

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = cv2.imread(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)

        return inp, kpt_2d, mask, path

    def get_training_img(self, img, kpt_2d, mask):
        pixel_num = np.sum(mask)

        train_img = cv2.imread(self.bg_paths[np.random.randint(len(self.bg_paths))])
        train_img = cv2.resize(train_img, (
        tless_config.train_w, tless_config.train_h))  # w = 720 H = 540 --> utils - tless - tless_config.py # 현재는 400*400 2020.04.15
        train_mask = np.zeros((tless_config.train_h, tless_config.train_w), dtype=np.uint8)
        tless_utils.cut_and_paste(img, mask, train_img, train_mask, 1) #random object 배치
        x_min, y_min, _, _ = cv2.boundingRect(mask)
        x, y, w, h = cv2.boundingRect(train_mask)
        kpt_2d = kpt_2d - [x_min, y_min] + [x, y]

        fused_img, fused_mask = tless_train_utils.get_fused_image(self.other_obj_coco, self.oann_ids, self.bg_paths)

        def paste_img0_on_img1(img0, mask0, img1, mask1):
            img, mask = img1.copy(), mask1.copy()
            mask_ = mask0 == 1
            img[mask_] = img0[mask_]
            mask[mask_] = 0
            return img, mask

        if np.random.uniform() < 0.5:
            train_img, _ = paste_img0_on_img1(train_img, train_mask, fused_img, fused_mask)
        else:
            img, mask = paste_img0_on_img1(fused_img, fused_mask, train_img, train_mask)
            if np.sum(mask) / pixel_num < cfg.tless.ratio:
                train_img, _ = paste_img0_on_img1(train_img, train_mask, fused_img, fused_mask)
            else:
                train_img, train_mask = img, mask

        x, y, w, h = cv2.boundingRect(train_mask)
        bbox = [x, y, x + w - 1, y + h - 1]

        return train_img, kpt_2d, train_mask, bbox

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]
        img, kpt_2d, mask, rgb_path = self.read_data(img_id)

        if self.split == 'train':
            # Add background
            # train_img = cv2.imread(self.bg_paths[np.random.randint(len(self.bg_paths))])
            # train_img = cv2.resize(train_img, (img.shape[1], img.shape[0]))
            # img, mask = cut_and_paste(img, mask, train_img)

            # rot = np.random.uniform() * 360.
            # img, _ = tless_train_utils.rotate_image(img, rot, get_rot=True)
            # if np.random.uniform() < 0.8:
            #     imgaug.seed(int(round(time.time() * 1000) % (2 ** 16)))
            #     img = tless_train_utils.color_jitter.augment_image(img)
            # mask, rot = tless_train_utils.rotate_image(mask, rot, get_rot=True)
            # kpt_2d = data_utils.affine_transform(kpt_2d, rot)

            img, kpt_2d, mask, bbox = self.get_training_img(img, kpt_2d, mask)
            # 202.04.17 shape: 540*720으로 바꿔서 test 진행 해봐야 함. 이미지 크기에 따른 차이가 존재 할 것같다는 추측
            # 실험목적: background + color jitter + rotation + crop & resize + +add other object (400 * 400)
            # 실험목적: background + color jitter + rotation + crop & resize + +add other object (540 * 720)
            # 예상으로는 540 * 720 일 때 더 좋을 것 같음 - (이유: 이미지 사이즈가 크면 더 많은 정보처리에 강해지기때문에)
            # print(img.shape) # 2020.04.15 - shape 400 * 400

            if np.random.uniform() < 0.8:
                imgaug.seed(int(round(time.time() * 1000) % (2 ** 16)))
                img = tless_train_utils.color_jitter.augment_image(img)

            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)

        else:
            inp = img

        # img_ = inp.copy()
        # cv2.circle(img_, (int(kpt_2d[0][0]), int(kpt_2d[0][1])), 5, (0, 0, 255), -1)
        # cv2.circle(img_, (int(kpt_2d[1][0]), int(kpt_2d[1][1])), 5, (255, 0, 0), -1)
        # cv2.circle(img_, (int(kpt_2d[2][0]), int(kpt_2d[2][1])), 5, (0, 255, 0), -1)
        # cv2.circle(img_, (int(kpt_2d[3][0]), int(kpt_2d[3][1])), 5, (255, 0, 255), -1)
        # cv2.circle(img_, (int(kpt_2d[4][0]), int(kpt_2d[4][1])), 5, (0, 255, 255), -1)
        # cv2.circle(img_, (int(kpt_2d[5][0]), int(kpt_2d[5][1])), 5, (255, 255, 0), -1)
        # cv2.circle(img_, (int(kpt_2d[6][0]), int(kpt_2d[6][1])), 5, (125, 0, 0), -1)
        # cv2.circle(img_, (int(kpt_2d[7][0]), int(kpt_2d[7][1])), 5, (0, 125, 0), -1)
        # cv2.circle(img_, (int(kpt_2d[8][0]), int(kpt_2d[8][1])), 5, (255, 255, 255), -1)
        # mask_ = mask.copy()
        # mask_ = np.where(mask_ > 0, 255, 0)
        # cv2.imwrite("./output/test_img/{}_output_mask.jpg".format(index), mask_)
        # cv2.imwrite("./output/test_img/{}_output.jpg".format(index), img_)
        
        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {'rgb_path': rgb_path}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask
