import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([init, se])
    return x
def conv_block(inputs, filters):
    x = inputs
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    return x
def encoder1(inputs):
    skip_connections = []
    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    output = model.get_layer("block5_conv4").output
    return output, skip_connections
 
def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)
    return x
# def encoder2(inputs):
#     skip_connections = []
#     output = DenseNet121(include_top=False, weights='imagenet')(inputs)
#     model = tf.keras.models.Model(inputs, output)
#
#     names = ["input_2", "conv1/relu", "pool2_conv", "pool3_conv"]
#     for name in names:
#         skip_connections.append(model.get_layer(name).output)
#     output = model.get_layer("pool4_conv").output
#
#     return output, skip_connections
def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs
    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)
    return x, skip_connections
def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)
    return x
def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x
def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)
def ASPP(x, filter):
    shape = x.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x) # ASPP
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    return y
def create_double_u_net(input_shape):
    inputs = Input(input_shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)
    x = inputs * outputs1
    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])
    model = Model(inputs, outputs)
    return model

def create_unet(input_shape):
    inputs = tf.keras.Input(input_shape)    
    conv1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv1)

    pool2 = MaxPool2D(pool_size=(2,2),padding='same')(conv2)
    conv3 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(pool2)
    conv4 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv3)

    pool4 = MaxPool2D(pool_size=(2,2),padding='same')(conv4)
    conv5 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(pool4)
    conv6 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(conv5)

    pool6 = MaxPool2D(pool_size=(2,2))(conv6)
    conv7 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(pool6)
    conv8 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(conv7)

    pool8 = MaxPool2D(pool_size=(2,2))(conv8)
    conv9 = Conv2D(1024, kernel_size=(3,3), activation='relu', padding='same')(pool8)
    conv10 = Conv2D(1024, kernel_size=(3,3), activation='relu', padding='same')(conv9)

    up11 = UpSampling2D(size=(2, 2))(conv10)
    up11 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(up11)
    up11 = Concatenate(axis=-1)([conv8,up11])
    conv12 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(up11)
    conv13 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(conv12)

    up14 = UpSampling2D((2,2))(conv13)
    up14 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(up14)
    up14 = Concatenate(axis=-1)([conv6,up14])
    conv15 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(up14)
    conv16 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(conv15)

    up17 = UpSampling2D((2,2))(conv16)
    up17 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(up17)
    up17 = Concatenate(axis=-1)([conv4,up17])
    conv18 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(up17)
    conv19 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv18)

    up18 = UpSampling2D((2,2))(conv19)
    up18 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(up18)
    up18 = Concatenate(axis=-1)([conv2,up18])
    conv19 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(up18)
    conv20 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv19)
    conv21 = Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding = 'same')(conv20)

    output = conv21
    model = tf.keras.Model(inputs,output)

    return output