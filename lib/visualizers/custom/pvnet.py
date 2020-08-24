from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.vsd import inout
import os
import cv2

from pytless import renderer

mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch, tless_cls, test_file):
        # Setting path
        save_path = os.path.join('./output', 'tless_obj {:02d}_test {:02d}'.format(tless_cls, test_file))
        mask_path = os.path.join(save_path, "mask")
        ren_rgb_path = os.path.join(save_path, "rgb_rendering")
        bbox_rgb_path = os.path.join(save_path, "rgb_bbox")

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(mask_path)
            os.mkdir(ren_rgb_path)
            os.mkdir(bbox_rgb_path)

        rgb_path = batch['meta']['rgb_path'][0]
        rgb_path_split = rgb_path.split('/')
        rgb = cv2.imread(rgb_path)

        # Loading model
        model_dir = 'data/tless/models_cad'
        obj_path = os.path.join(model_dir, 'obj_{:02d}.ply'.format(tless_cls))
        model = inout.load_ply(obj_path)

        # Saving output mask
        output_mask = output['mask'][0].detach().cpu().numpy()
        output_mask = np.where(output_mask>0, 255, 0).astype('uint8')
        im_size = (output_mask.shape[1], output_mask.shape[0])
        cv2.imwrite(os.path.join(mask_path, rgb_path_split[3]), output_mask)

        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])
        pose_gt = np.array(anno['pose'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        R = pose_pred[:, :3]
        t = pose_pred[:, 3] * 1000

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        # Rendering image using output pose
        vis_rgb = np.zeros(rgb.shape, np.float)
        surf_color = (0, 0, 255)
        ren_rgb = renderer.render(model, im_size, K, R, t,
                                  surf_color=surf_color, mode='rgb')
        vis_rgb += 0.7 * ren_rgb.astype(np.float)
        vis_rgb = 0.8 * rgb + 0.2 * vis_rgb
        vis_rgb[vis_rgb > 255] = 255
        vis_rgb = vis_rgb.astype(np.uint8)
        cv2.imwrite(os.path.join(ren_rgb_path, rgb_path_split[3]), vis_rgb)

        vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)

        _, ax = plt.subplots(1)
        ax.imshow(vis_rgb)
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))

        ax.add_patch(patches.Circle((kpt_2d[0][0],kpt_2d[0][1]), radius= 5, color='r'))
        ax.add_patch(patches.Circle((kpt_2d[1][0],kpt_2d[1][1]), radius= 5, color='b'))
        ax.add_patch(patches.Circle((kpt_2d[2][0],kpt_2d[2][1]), radius= 5, color='darkviolet'))
        ax.add_patch(patches.Circle((kpt_2d[3][0],kpt_2d[3][1]), radius= 5, color='orange'))
        ax.add_patch(patches.Circle((kpt_2d[4][0],kpt_2d[4][1]), radius= 5, color='cyan'))
        ax.add_patch(patches.Circle((kpt_2d[5][0],kpt_2d[5][1]), radius= 5, color='y'))
        ax.add_patch(patches.Circle((kpt_2d[6][0],kpt_2d[6][1]), radius= 5, color='yellowgreen'))
        ax.add_patch(patches.Circle((kpt_2d[7][0],kpt_2d[7][1]), radius= 5, color='g'))
        ax.add_patch(patches.Circle((kpt_2d[8][0],kpt_2d[8][1]), radius= 5, color='k'))

        plt.savefig(os.path.join(bbox_rgb_path, "{}".format(rgb_path_split[3])))

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)





