import math
import numpy as np
from scipy.spatial.transform import Rotation as R


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler anglesfgsm
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

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


if __name__ =="__main__":
    x_angle = math.atan2(-0.708838, -0.671032)
    z_angle = math.atan2(0.140804, -0.965872)
    print("x angle is {} \nz angle is {}".format(x_angle, z_angle))

    pose_path = 'data/linemod/driller/pose/pose2.npy'
    pose = np.load(pose_path)

    pose = pose[:,:3]
    print(pose)
    pose_angle = rotationMatrixToEulerAngles(pose)
    print(pose_angle)
    print("="*30)
    R = eul2rot(pose_angle)
    # x = R.from_euler('zyx', [pose_angle[2],pose_angle[1],pose_angle[0]], degrees=True)
    # y = R.from_euler('y', pose_angle[1], degrees=True)
    # z = R.from_euler('z', pose_angle[2], degrees=True)
    # x_pose = x.as_matrix()
    # y_pose = y.as_matrix()
    # z_pose = z.as_matrix()
    # a = x_pose + y_pose + z_pose
    print(R)