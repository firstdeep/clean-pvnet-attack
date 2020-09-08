import os
import numpy as np
import tqdm
from lib.utils.vsd import inout
from pytless import renderer
import yaml
import cv2
from transforms3d.euler import mat2euler, euler2mat
import math

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

def blender_pose_to_blender_euler(pose):
    euler = mat2euler(pose, axes='szyx')
    return np.array(euler)


def blender_euler_to_blender_pose(euler):
    azi = euler[0]
    ele = euler[1]
    theta = euler[2]
    pose = euler2mat(azi, ele, theta, axes='szyx')
    return pose

def rot2eul(R):

    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def r2d(x):
    return x * 180.0 / np.pi

def d2r(x):
    return x * np.pi / 180.0

def get_mask():

    data_root = './data/linemod/driller'
    obj_path = os.path.join(data_root, 'driller.ply')
    model = inout.load_ply(obj_path)
    rgb_dir = os.path.join(data_root, 'JPEGImages')
    mask_dir = os.path.join(data_root, 'mask_re')
    pose_dir = os.path.join(data_root, 'pose')

    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    inds = range(len(os.listdir(rgb_dir)))
    model['pts'] *= 1000.

    #for ind in tqdm.tqdm(inds):
    for ind in tqdm.tqdm(inds):
        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)

        #rotation 변경
        pose_rotation  = pose[:,:3]

        pose_angel_ro, _ = cv2.Rodrigues(pose_rotation)
        # X 축으로 5도 이동
        pose_angel_ro[0] += d2r(10)
        # pose_angel_ro[1] += d2r(7)
        # pose_angel_ro[2] += d2r(5)

        pose_ro, _ = cv2.Rodrigues(pose_angel_ro)

        pose[:,:3] = pose_ro

        #translation 변경
        # pose[0, 3] += 0.02  # 30mm 이동
        # pose[1, 3] += 0.02  # 30mm 이동

        rgb_path = os.path.join(rgb_dir, '{:06d}.jpg'.format(ind))
        rgb = cv2.imread(rgb_path)
        im_size = (rgb.shape[1], rgb.shape[0])

        K = linemod_K
        R = pose[:, :3]
        t = pose[:, 3:] * 1000.

        surf_color = (255, 255, 0)
        ren_rgb = renderer.render(model, im_size, K, R, t, clip_near=10, clip_far=10000,
                                  surf_color=surf_color, mode='rgb')
        gray = cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float)
        vis_gray = np.where(gray > 0, 255, 0)
        cv2.imwrite(mask_dir + "/{:04d}.png".format(ind), vis_gray.astype(np.uint8))

if __name__=="__main__":
    get_mask()
    # A = cv2.imread("/home/hwanglab/PycharmProjects/clean-pvnet-custom/data/linemod/driller/mask_o/0020.png")
    # B = cv2.imread("/home/hwanglab/PycharmProjects/clean-pvnet-custom/data/linemod/driller/mask/0020.png")
    # add = cv2.add(A,B)
    # cv2.imshow("imag",add)
    # cv2.waitKey(100000)