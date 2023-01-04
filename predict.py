from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
import cv2
from loader import load_test_data
from util import yolo_voc

x, y = load_test_data(1)
px = keras.applications.resnet50.preprocess_input(np.copy(x))

model = load_model("split.h5")
pred = model.predict(np.expand_dims(px[0], axis=0))
print("pred", pred[0])
print("y", y[0])

h, w, _ = x[0].shape
(sx, sy, ex, ey) = yolo_voc(pred[0], (w, h))
(s2x, s2y, e2x, e2y) = yolo_voc(y[0], (w, h))
im = x[0].astype(np.uint8)
cv2.rectangle(im, (sx, sy), (ex, ey), (255, 0, 0), 1)
cv2.rectangle(im, (s2x, s2y), (e2x, e2y), (0, 255, 0), 1)

cv2.imshow("out", im)
cv2.waitKey(0)
