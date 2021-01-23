from utils import transform_name, transform_name_for_x_train, transform_name_for_y_train, ground_truth_folder, training_folder
from model import build_model
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data import data_squence
from metrics import DiceLoss, iou
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
import pathlib
warnings.filterwarnings('ignore')


filename = os.listdir('validation/mask')
x_test = list(map(lambda a: 'validation/image/' +
                  a.replace('_segmentation.png', '.jpg'), filename))
y_test = list(map(lambda a: 'validation/mask/'+a, filename))

file_name = os.listdir('data_augmented/mask')
x_train = list(map(transform_name_for_x_train, file_name))
y_train = list(map(transform_name_for_y_train, file_name))

train = data_squence(x_train, y_train, batch_size=16,)
validation = data_squence(x_test, y_test, batch_size=16, )


pathlib.Path("./checkpoint").mkdir(exist_ok=True, parents=True)


mc = ModelCheckpoint(filepath=os.path.join(
    './checkpoint', "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir='./log', write_graph=True)
model = build_model((192, 256, 3))
loss_function = DiceLoss()
model.compile('Adam', loss=loss_function, metrics=[iou])
model.fit(train, epochs=100, validation_data=validation,
          initial_epoch=0,
          verbose=1,
          callbacks=[tb, mc])
