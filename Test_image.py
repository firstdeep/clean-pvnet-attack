import numpy as np
import cv2

if __name__ == "__main__":
    A = cv2.imread("/home/hwanglab/PycharmProjects/clean-pvnet-custom/data/linemod/driller/JPEGImages/000000.jpg")
    B = cv2.imread("/home/hwanglab/PycharmProjects/clean-pvnet-custom/data/linemod/driller/JPEGImagesP/000000.jpg")

    cv2.rectangle(B, (500, 400),  (600, 450), (0, 0, 255), 3);
    sub = np.subtract(B,A)
    cv2.imwrite("/home/hwanglab/PycharmProjects/clean-pvnet-custom/data/linemod/driller/JPEGImagesP/000000.jpg", B)
    # cv2.imshow("sub_image", B)
    # cv2.waitKey(0)