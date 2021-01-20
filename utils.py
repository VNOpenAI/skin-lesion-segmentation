import numpy as np
import cv2
def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    return image, mask

def transform_name(string_):
    result = string_.replace('.jpg','_segmentation.png')
    return result

