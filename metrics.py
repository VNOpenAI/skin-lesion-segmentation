import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Flatten

smooth = 1e-152
gamma =2
alpha = 0.6
class DiceLoss(tf.keras.loss.Loss):
    def call(self,y_true,y_pred):
        y_true = Flatten()(y_true)
        y_pred = Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true*y_pred)
        return 1 - ((2*intersection +smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) +smooth))    

"""def focal_loss(y_true,y_pred):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    loss = alpha * y_true * """
