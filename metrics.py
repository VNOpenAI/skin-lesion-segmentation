import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.backend import epsilon

gamma = 2
alpha = 0.6


class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = Flatten()(y_true)
        y_pred = Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true*y_pred)
        return 1 - ((2*intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))


"""def focal_loss(y_true,y_pred):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    loss = alpha * y_true * """


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + epsilon) / (union + epsilon)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
