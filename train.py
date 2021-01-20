from sklearn.utils.validation import check_random_state
from utils import transform_name
from model import build_model
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data import data_squence
from metrics import DiceLoss
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')
ground_truth_folder = 'ISIC2018_task1_GroundTruth'
training_folder = 'ISIC2018_task1_training'

x = os.listdir(training_folder)
y = list(map(transform_name, x))
x = list(map(lambda a: training_folder + '/'+a, x))
y = list(map(lambda a: ground_truth_folder + '/'+a, y))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

train = data_squence(x_train, y_train, batch_size=16, aug=True)
validation = data_squence(x_test, y_test, batch_size=16, aug=False)
mc = ModelCheckpoint(filepath=os.path.join(
    './checkpoint', "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir='./log', write_graph=True)
model = build_model((384, 512, 3))
loss_function = DiceLoss()
model.compile('Adam', loss=loss_function)
model.fit(train,epochs =3,validation_data = validation,
            initial_epoch=0,
            verbose=1,           
            callbacks=[tb,mc])
