import numpy as np
import cv2

ground_truth_folder = 'ISIC2018_task1_GroundTruth'
training_folder = 'ISIC2018_task1_training'

def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    return image, mask

def transform_name(string_):
    result = string_.replace('.jpg','_segmentation.png')
    return result

def transform_name_for_x_train(string_):
    result = 'data_augmented/images/'+string_.replace('segmentation_','')
    return result 

def transform_name_for_y_train(string_):
    result = 'data_augmented/mask/' + string_
    return result
