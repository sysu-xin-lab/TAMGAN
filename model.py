import numpy as np
from tensorflow.python.keras.layers import *
from tensorflow import keras
import tensorflow as tf
from functools import reduce
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K

num_images = 46


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


# fundamental 3D conv_block
def conv_block(input_tensor, filters, kernel_size=(3, 3, 3), padding='valid'):
    output_tensor = compose(
        keras.layers.Conv3D(filters, kernel_size, padding=padding),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.ReLU())(input_tensor)
    return output_tensor

def discriminator_block(input_tensor, filters, kernel_size=(3, 3, 3), stride=1, padding='same'):
    output_tensor = compose(
        keras.layers.Conv3D(filters, kernel_size, strides=stride, padding=padding), 
        keras.layers.BatchNormalization(momentum=0.8), 
        keras.layers.LeakyReLU(alpha=0.2))(input_tensor) 
    return output_tensor

### 3D res_block ###
def residual_block(x, kernel, filters=16):
    x = Conv3D(filters=filters, kernel_size=(1, 1, 1), padding='same')(x)
    x1 = BatchNormalization()(x)
    x1 = ReLU()(x1)
    x1 = Conv3D(filters=filters, kernel_size=kernel, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv3D(filters=filters, kernel_size=kernel, padding='same')(x1)
    x = x1 + x
    x = ReLU()(x)
    return x


def tam_block(input_feature, ratio=2):
    channel_axis = 1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D(data_format='channels_first')(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D(data_format='channels_first')(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    cbam_feature = Reshape((channel, 1, 1, 1))(cbam_feature)
    return multiply([input_feature, cbam_feature])


### model difination ###
def generate_model(input_shape):
    data_tensor = Input(input_shape, name='input1')  # define input layer

    ### multi-scale encoding ###
    l1 = conv_block(data_tensor, filters=16, kernel_size=(3, 3, 3), padding='same')  # 3×3feature extraction

    l2 = MaxPool3D(pool_size=(1, 2, 2))(l1)

    l3 = MaxPool3D(pool_size=(1, 4, 4))(l1)

    l3_1 = residual_block(l3, kernel=(3, 3, 3), filters=16)
    l3_2 = residual_block(l3_1, kernel=(3, 3, 3), filters=16)
    l3_3 = residual_block(l3_2, kernel=(3, 3, 3), filters=16)
    l3_4 = residual_block(l3_3, kernel=(3, 3, 3), filters=16)
    l3_5 = residual_block(l3_4, kernel=(3, 3, 3), filters=16)
    l3 = concatenate([l3_1, l3_3, l3_5], axis=-1)

    l3_up = UpSampling3D(size=(1, 2, 2))(l3)
    l2 = concatenate([l2, l3_up], axis=-1)
    l2_1 = residual_block(l2, kernel=(3, 3, 3), filters=16)
    l2_2 = residual_block(l2_1, kernel=(3, 3, 3), filters=16)
    l2_3 = residual_block(l2_2, kernel=(3, 3, 3), filters=16)
    l2_4 = residual_block(l2_3, kernel=(3, 3, 3), filters=16)
    l2_5 = residual_block(l2_4, kernel=(3, 3, 3), filters=16)
    l2 = concatenate([l2_1, l2_3, l2_5], axis=-1)

    l3_up2 = UpSampling3D(size=(1, 2, 2))(l3_up)
    l2_up = UpSampling3D(size=(1, 2, 2))(l2)
    l1 = concatenate([l1, l2_up, l3_up2], axis=-1)
    l1_1 = residual_block(l1, kernel=(3, 3, 3), filters=16)
    l1_2 = residual_block(l1_1, kernel=(3, 3, 3), filters=16)
    l1_3 = residual_block(l1_2, kernel=(3, 3, 3), filters=16)
    l1_4 = residual_block(l1_3, kernel=(3, 3, 3), filters=16)
    l1_5 = residual_block(l1_4, kernel=(3, 3, 3), filters=16)
    l1 = concatenate([l1_1, l1_3, l1_5], axis=-1)

    ### temporal attention ###
    l3_a = tam_block(l3, 2)
    l3_a_up = UpSampling3D(size=(1, 2, 2))(l3_a)
    l2_a = concatenate([l2, l3_a_up], axis=-1)
    l2_a = tam_block(l2_a, 2)
    l3_a_up2 = UpSampling3D(size=(1, 2, 2))(l3_a_up)
    l2_a_up = UpSampling3D(size=(1, 2, 2))(l2_a)
    l1_a = concatenate([l1, l2_a_up, l3_a_up2], axis=-1)
    l1_a = tam_block(l1_a, 2)

    x = concatenate([l1_a, l2_a_up, l3_a_up2], axis=-1)
    ### decoding ###
    x = conv_block(x, filters=128, kernel_size=(3, 3, 3), padding='same')
    x = conv_block(x, filters=64, kernel_size=(3, 3, 3), padding='same')
    x = conv_block(x, filters=16, kernel_size=(3, 3, 3), padding='same')
    x = conv_block(x, filters=4, kernel_size=(3, 3, 3), padding='same')
    model = keras.Model(data_tensor, x, name='generation_model')
    return model

def discriminate_model(input_shape):
    data_tensor = Input(input_shape, name='input2')
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(data_tensor)
    x = LeakyReLU(alpha=0.2)(x)

    x = discriminator_block(x, filters=64, stride=2)
    x = discriminator_block(x, filters=128, stride=1)
    x = discriminator_block(x, filters=128, stride=2)
    x = discriminator_block(x, filters=256, stride=1)
    x = discriminator_block(x, filters=256, stride=2)
    x = discriminator_block(x, filters=512, stride=1)
    x = discriminator_block(x, filters=512, stride=2)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = keras.Model(data_tensor, x, name='discrimination_model')
    return model

def get_gan_network(input_shape, model_g, model_d, optimizer, loss):
    model_d.trainable = False
    gan_input = Input(shape=input_shape)
    x = model_g(gan_input)
    gan_output = model_d(x)
    gan = keras.Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[loss, "binary_crossentropy"], loss_weights=[1., 0.1], optimizer=optimizer)
    return gan


### loss function ###
def loss_fun(y_true, y_pred):
    glo_loss = tf.reduce_mean((y_true[:, :, :, :, 0:4] - y_pred) ** 2)
    anti_mask = 1 - y_true[:, :, :, :, 4:8]
    local_loss = tf.reduce_mean((y_true[:, :, :, :, 0:4] * anti_mask - y_pred * anti_mask) ** 2)
    speloss = glo_loss + local_loss
    # ssimloss = 1 - tf.image.ssim(y_true[:, :, :, :, 0:4], y_pred, max_val=1.0)
    final_loss = speloss * 10000
    return final_loss
