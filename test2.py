import numpy as np
import cv2
fimg1="in.jpg"
fimg2="out.png"
img=cv2.imread(fimg1, 0)
mask = img < 128
img=cv2.imread(fimg1)
nimg = np.zeros( (img.shape[0], img.shape[1], 4), np.uint8 )
nimg[:, :, :3] = img
nimg[:, :, 3] = mask.astype(np.uint8) * 255

cv2.imwrite(fimg2, nimg)

