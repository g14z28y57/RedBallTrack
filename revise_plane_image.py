import cv2
from IPython import embed
import numpy as np
import os

img_path = "plane_origin.jpg"
assert os.path.exists(img_path)
img = cv2.imread(img_path)
img = np.maximum(img - 50, 0)
cv2.imwrite("plane.jpg", img)
