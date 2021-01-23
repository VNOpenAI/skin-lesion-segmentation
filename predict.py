import tensorflow as tf
import numpy as np
from data import get_predict
import cv2
import os
import sklearn.preprocessing
file_path_model = 'model.h5'
trained_model = tf.keras.models.load_model(file_path_model, compile=False)
size = (512, 384)
images_folder = 'C:/Users/chinhdv2/Desktop/double-Unet/test'
file_img = os.listdir(images_folder)
cv2.namedWindow('a', cv2.WINDOW_AUTOSIZE)
for index, i in enumerate(file_img):
    name = images_folder + '/'+i
    img2 = cv2.imread(name, cv2.IMREAD_COLOR)
    out_put = get_predict(trained_model, img2)
    out_put = out_put.astype(np.uint8)
    out_put = np.stack([out_put, out_put, out_put], axis=-1)
    out_put2 = np.concatenate([cv2.resize(img2, size), out_put], axis=1)
    cv2.imshow('a', out_put2)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
cv2.destroyAllWindows()
