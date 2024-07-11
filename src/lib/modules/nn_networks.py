# Standar imports
import sys

# Project imports
from ..modules import tf_utils
from ..modules.nn_layers import *
from ..modules import operators

# 3rd party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, Constant

from tensorflow.keras.layers import Layer, Conv3D, Conv2D, Activation, Input, Concatenate, Conv3DTranspose, Dense, \
    BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Reshape, Flatten, GaussianNoise, AveragePooling3D, Lambda, ReLU, \
    MaxPooling3D
from tensorflow.keras.layers import UpSampling2D, UpSampling3D

def generator_net(shape, redux=False, unit_domain=True, ndims=2):
    if unit_domain:
        act = "linear"
    # act = "tanh"
    else:
        act = "linear"
    # act ="tanh"

    max_pool = getattr(KL, 'MaxPooling%dD' % ndims)
    conv = getattr(KL, 'Conv%dD' % ndims)
    convTranspose = getattr(KL, 'Conv%dDTranspose' % ndims)
    concat_1 = Concatenate(axis=1)
    concat_2 = Concatenate(axis=1)
    concat_3 = Concatenate(axis=1)
    concat_4 = Concatenate(axis=1)
    concat_5 = Concatenate(axis=1)
    same_conv_1 = same_conv_layer(24, 3, 1, 'same', "s_1", 1, ndims)
    same_conv_2 = same_conv_layer(32, 3, 1, 'same', "s_2", 1, ndims)
    same_conv_3 = same_conv_layer(32, 3, 1, 'same', "s_3", 1, ndims)
    same_conv_4 = same_conv_layer(64, 3, 1, 'same', "s_4", 1, ndims)
    same_conv_5 = same_conv_layer(48, 3, 1, 'same', "s_5", 1, ndims)
    same_conv_6 = same_conv_layer(96, 3, 1, 'same', "s_6", 1, ndims)
    same_conv_7 = same_conv_layer(64, 3, 1, 'same', "s_7", 1, ndims)
    same_conv_8 = same_conv_layer(64, 3, 1, 'same', "s_8", 1, ndims)
    same_conv_9 = same_conv_layer(48, 3, 1, 'same', "s_9", 1, ndims)
    same_conv_10 = same_conv_layer(32, 3, 1, 'same', "s_10", 1, ndims)
    same_conv_11 = same_conv_layer(24, 3, 1, 'same', "s_11", 1, ndims)
    dil_conv_1 = same_conv_layer(32, 3, 1, 'same', "di_1", 2, ndims)
    dil_conv_2 = same_conv_layer(48, 3, 1, 'same', "di_2", 2, ndims)
    end_conv = conv(ndims, 3, strides=1, padding='same', name="final_convolution",
                    kernel_initializer=RandomNormal(mean=0.0, stddev=1E-6), bias_initializer=Constant(value=0),
                    activation=act, data_format="channels_first")
    raw_conv_1 = conv(24, 2, strides=1, padding='same', name="d_1", data_format="channels_first")
    raw_conv_2 = conv(32, 2, strides=1, padding='same', name="d_2", data_format="channels_first")
    raw_conv_3 = conv(32, 2, strides=1, padding='same', name="d_3", data_format="channels_first")
    raw_conv_4 = conv(48, 2, strides=1, padding='same', name="d_4", data_format="channels_first")
    pool_1 = max_pool(2, padding="valid", name="p_1", data_format="channels_first")
    pool_2 = max_pool(2, padding="valid", name="p_2", data_format="channels_first")
    pool_3 = max_pool(2, padding="valid", name="p_3", data_format="channels_first")
    pool_4 = max_pool(2, padding="valid", name="p_4", data_format="channels_first")
    pool_redux = max_pool(2, padding="valid", name="p_red", data_format="channels_first")
    up_conv_1 = transpose_convolution(48, 2, 2, "same", "up_1", ndims)
    up_conv_2 = transpose_convolution(32, 2, 2, "same", "up_2", ndims)
    up_conv_redux = transpose_convolution(32, 2, 2, "same", "up_redux", ndims)

    def f(F, M):
        x = concat_1([F, M])
        x = same_conv_1(x)
        if redux:
            x = pool_redux(x)
        high = raw_conv_1(x)
        high = pool_1(high)
        high = dil_conv_1(high)
        res_1 = high
        high = raw_conv_2(high)
        high = pool_2(high)
        res_2 = dil_conv_2(high)
        mid = same_conv_2(x)
        mid = same_conv_3(mid)
        low = raw_conv_3(mid)
        low = pool_3(low)
        low = concat_2([low, res_1])
        low = same_conv_4(low)
        low = same_conv_5(low)
        lowest = raw_conv_4(low)
        lowest = pool_4(lowest)
        lowest = concat_3([lowest, res_2])
        lowest = same_conv_6(lowest)
        lowest = same_conv_7(lowest)
        lowest = up_conv_1(lowest)
        low = concat_4([lowest, low])
        low = same_conv_8(low)
        low = same_conv_9(low)
        low = up_conv_2(low)
        mid = concat_5([low, mid])
        mid = same_conv_10(mid)
        if redux:
            mid = up_conv_redux(mid)
        mid = same_conv_11(mid)
        output = end_conv(mid)
        # if ndims == 3:
        #     output.set_shape((None, shape[0], shape[1], shape[2], 3))
        # if ndims == 2:
        #     output.set_shape((None, shape[0], shape[1], 2))
        return output

    return f


# Discriminatro arquitecture from Luwen Duan
def discriminator_net(redux=False, ndims=2):
    conv = getattr(KL, 'Conv%dD' % ndims)
    if redux:
        fact = 4
    else:
        fact = 1
    raw_conv_1 = conv(24, 2, strides=1, padding='same', name="d_1", data_format="channels_first")

    bn_conv_1 = bn_conv_layer(24 // fact, 5, 1, 'same', "bn_1", True, ndims)
    bn_conv_2 = bn_conv_layer(24 // fact, 5, 1, 'same', "bn_2", True, ndims)
    bn_conv_3 = bn_conv_layer(32 // fact, 5, 1, 'same', "bn_3", True, ndims)
    bn_conv_4 = bn_conv_layer(48 // fact, 5, 1, 'same', "bn_4", True, ndims)
    bn_conv_5 = bn_conv_layer(32 // fact, 3, 1, 'same', "bn_5", True, ndims)

    fc_1 = Dense(128 // (fact ** 2), name="fc_1")
    fc_2 = Dense(32 // fact, name="fc_2")
    fc_3 = Dense(1, name="fc_end", activation='linear')
    concat = Concatenate(axis=1)
    flat = Flatten()

    def f(F, M):
        x = concat([F, M])
        x = bn_conv_1(x)
        x = bn_conv_2(x)
        x = bn_conv_3(x)
        x = bn_conv_4(x)
        x = bn_conv_5(x)
        x = flat(x)
        x = fc_1(x)
        x = fc_2(x)
        x = fc_3(x)
        return x

    return f


def downsampler():
    pool_layer_1 = Conv2D(1, 2, strides=2, use_bias=False, name='downsample_1')
    pool_layer_2 = Conv2D(1, 2, strides=2, use_bias=False, name='downsample_2')

    def f(input):
        down_1 = pool_layer_1(input)
        down_2 = pool_layer_2(down_1)
        pool_layer_1.set_weights([np.zeros((2, 2, 1, 1)) + 0.25])
        pool_layer_2.set_weights([np.zeros((2, 2, 1, 1)) + 0.25])
        pool_layer_1.trainable = False
        pool_layer_2.trainable = False
        return down_1, down_2

    return f


