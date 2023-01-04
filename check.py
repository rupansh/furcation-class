import numpy as np
import cv2
from loader import load_test_data
from util import yolo_voc

x, y = load_test_data(1)

print("y", y[0])

im = x[0].astype(np.uint8)
h, w, _ = im.shape
print(w, h)
l, t, r, b = yolo_voc(y[0], (w, h))
cv2.rectangle(im, (l, t), (r, b), (0, 255, 0), 1)

cv2.imshow("out", im)
while cv2.waitKey(0) != 27:
    continue
