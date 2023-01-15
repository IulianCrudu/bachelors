import numpy as np
import cv2 as cv

from utils import IMG_SIZE


def dense_optical_flow(path1: str, path2: str, debug: bool = False):
    # BGR format
    img1 = cv.imread(
        path1, cv.IMREAD_COLOR
    )
    img1 = cv.resize(img1, IMG_SIZE, interpolation=cv.INTER_LINEAR)
    frame1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.imread(
        path2, cv.IMREAD_COLOR
    )
    img2 = cv.resize(img2, IMG_SIZE, interpolation=cv.INTER_LINEAR)
    frame2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255

    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang*180/np.pi/2

    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    if debug:
        cv.imshow('frame2', bgr)
        cv.waitKey(0)

    cv.imwrite('opticalfb.png', frame2)
    cv.imwrite('opticalhsv.png', bgr)
    return flow


if __name__ == "__main__":
    dense_optical_flow()


