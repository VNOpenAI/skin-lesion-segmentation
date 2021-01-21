from sklearn.utils.validation import check_random_state
from utils import transform_name, transform_name_for_x_train,transform_name_for_y_train
from model import build_model
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data import data_squence
from metrics import DiceLoss
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
from utils import ground_truth_folder,training_folder
warnings.filterwarnings('ignore')

x = os.listdir(training_folder)
y = list(map(transform_name, x))
x = list(map(lambda a: training_folder + '/'+a, x))
y = list(map(lambda a: ground_truth_folder + '/'+a, y))

_, x_test, _, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

file_name = os.listdir('data_augmented/mask')
x_train = list(map(transform_name_for_x_train,file_name))
y_train = list(map(transform_name_for_y_train,file_name))

train = data_squence(x_train, y_train, batch_size=16,)
validation = data_squence(x_test, y_test, batch_size=16, )



mc = ModelCheckpoint(filepath=os.path.join(
    './checkpoint', "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir='./log', write_graph=True)
model = build_model((192, 256, 3))
loss_function = DiceLoss()
model.compile('Adam', loss=loss_function)
model.fit(train,epochs =3,validation_data = validation,
            initial_epoch=0,
            verbose=1,           
            callbacks=[tb,mc])
