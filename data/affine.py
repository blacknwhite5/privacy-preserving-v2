"""https://zhuanlan.zhihu.com/p/29515986"""
from skimage import transform as trans
import numpy as np
import cv2

src = np.array([
 [30.2946, 51.6963],
 [65.5318, 51.5014],
 [48.0252, 71.7366],
 [33.5493, 92.3655],
 [62.7299, 92.2041] ], dtype=np.float32 )


tform = trans.SimilarityTransform()
def affine_transform(img, landm, resize=(128,128)):
    dst = np.array(landm).reshape((5, 2))
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    img = cv2.imread(img)
    return cv2.warpAffine(img, M, resize, borderValue = 0.0)

