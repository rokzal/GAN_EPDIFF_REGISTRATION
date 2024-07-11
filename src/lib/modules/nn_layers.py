# Standar imports
import sys

# Project imports
from ..modules import tf_utils
from ..modules.operators import *

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


class SpatialTransformer(KL.Layer):
    # Spatial transformer layer. Applies deformation to image.
    def __init__(self, batch_size=1, unit_domain=False, dtype=tf.float32, custom_grad =False, param_layer = None, **kwargs):
        self.batch_size = batch_size
        self.unit_domain = unit_domain
        self.dtype_own = dtype
        self.custom_grad = custom_grad
        self.param_layer = param_layer
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dims = len(input_shape[0]) - 2

        self.im_shape = input_shape[0]
        shape_grid = list(self.im_shape[1:-1])

        self.grid = tf_utils.generate_grid(shape_grid, self.unit_domain, self.dtype_own)

    def call(self, inputs):
        # In trial
        # inputs[0] = tf.cast(inputs[0], "float16")
        # inputs[1] = tf.cast(inputs[1], "float16")
        if self.custom_grad:
            return tf.map_fn(self.interpn_call_custGrad, [inputs[0], inputs[1]], fn_output_signature=inputs[0].dtype)
        else:
            return tf.map_fn(self.interpn_call, [inputs[0], inputs[1]], fn_output_signature=inputs[0].dtype)

    # @tf.custom_gradient
    def interpn_call(self, inputs):
        #     def grad(upstream):
        #         return tf.ones_like(inputs[0]) * upstream, tf.ones_like(inputs[1]) * upstream

        return tf_utils.interpn(inputs[0], inputs[1] + self.grid)  # , grad

    @tf.custom_gradient
    def interpn_call_custGrad(self, inputs):
        res = tf_utils.interpn(inputs[0], inputs[1] + self.grid)
        #res = tf.constant(res)
        def grad(upstream):

            #debug
            # return tf.ones_like(inputs[0]), tf.ones_like(inputs[1])
            ndims = self.param_layer.ndims
            if ndims == 2:
                gx = tf_utils.gradientX(res[..., 0], True, self.param_layer)
                gy = tf_utils.gradientY(res[..., 0], True, self.param_layer)


                #gx = tf_utils.gradientX(res[np.newaxis,..., 0], False, self.param_layer)[0,...]
                #gy = tf_utils.gradientY(res[np.newaxis,..., 0], False, self.param_layer)[0,...]

                U0_1 = upstream[..., 0] * gx
                U0_2 = upstream[..., 0] * gy

                #Necesary since we are cheating the gradients, the lambda layer that changes to spatial coordinates can't
                #be skiped so we have to undo its effects on the gradients.
                fix = list(U0_1.shape)
                U0_2 /= (fix[1])
                U0_1 /= (fix[0])

                #Maybe need to change x and y coordinates. maybe do later?
                U = tf.stack([U0_2, U0_1], axis=-1)
                #U = tf.stack([U0_1, U0_2], axis=-1)

            elif ndims == 3:
                gx = tf_utils.gradientX(res[..., 0], True, self.param_layer)
                gy = tf_utils.gradientY(res[..., 0], True, self.param_layer)
                gz = tf_utils.gradientZ(res[..., 0], True, self.param_layer)

                U0_1 = upstream[..., 0] * gx
                U0_2 = upstream[..., 0] * gy
                U0_3 = upstream[..., 0] * gz

                #Same???
                fix = list(U0_1.shape)
                U0_2 /= (fix[1])
                U0_1 /= (fix[0])
                U0_3 /= (fix[2])

                U = tf.stack([U0_2, U0_1, U0_3], axis=-1)

            #-U : Important when w = m - u, since we change symbol earlier for spatial transformation (easy hack) but it should
            #Be unchanged for gradient calculation.
            return tf.ones_like(inputs[0]) * upstream, -U

        return res , grad


class IntegrateVector(KL.Layer):
    # Vector integration layer. Given a velocity field with stationary parameterization, integrates it s steps forward using
    # the scaling and squaring method.
    def __init__(self, s=8, batch_size=1, unit_domain=True, dtype=tf.float32, **kwargs):
        self.batch_size = batch_size
        self.unit_domain = unit_domain
        self.dtype_own = dtype
        self.s = s
        super(IntegrateVector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dims = len(input_shape[0]) - 2
        self.vel_shape = input_shape[0][1:]

    def call(self, inputs):
        # In trial
        # inputs[0] = tf.cast(inputs[0], "float16")
        # print("IntegrateVector" + str(inputs[0].dtype))

        return tf.map_fn(self.ss_integrate_call, inputs[0], fn_output_signature=inputs[0].dtype)
        # return tf_utils.ss_integrate_call(inputs[0], self.s)

    def ss_integrate_call(self, inputs):
        #return tf_utils.ss_integrate(inputs, self.s)
        return tf_utils.ss_integrate_tf(inputs, self.s)


class Stack(Layer):

    def __init__(self, dtype=tf.float32, **kwargs):
        self.dtype = dtype
        super(Stack, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Stack, self).build(input_shape)

    def call(self, x, training=None):
        res = K.stack(x, axis=4)[..., 0]
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0][0:4] + (3,)


class finiteDiffReg(Layer):
    # Custom layer, applies finiteDiff L operator for regularization on SVF's.

    def __init__(self, inverse=False, sigma=1.0, cn_gamma=1.0, cn_alpha=0.0025, s=1, gamma=0.0, shape=None,
                 batch_size=1, operator='lo', ndims=2, **kwargs):
        self.ndims = ndims
        self.inverse = inverse
        self.shape = shape
        self.ndims = ndims
        if operator == 'lo':
            op_class = CauchyNavierRegularizer(sigma, cn_gamma, cn_alpha, s, gamma, shape, batch_size, ndims)
            self.KK = op_class.KK
            if inverse:
                self.op = 1 / self.KK
            else:
                self.op = self.KK
        elif operator == 'fdnu':
            op_class = CNFiniteDifferecnesNotUnitDomain(sigma, cn_gamma, cn_alpha, s, gamma, shape=shape, ndims=ndims)
            self.KK = op_class.KK
            if inverse:
                self.op = 1 / self.KK
            else:
                self.op = self.KK

        elif operator == 'py':
            op_class = PycaOperator(shape[0], shape[1], shape[2], 0.0025, 2, 1, cn_alpha, 1)

        self.operator = operator
        self.op_class = op_class

        super(finiteDiffReg, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'inverse': self.inverse,
            'sigma': 1.0,
            'cn_gamma': 1.0,
            'cn_alpha': 0.0025,
            's': 2,
            'gamma': 1,
            'shape': self.shape,
            'batch_size': 1,
            'operator': 'lo',
            'ndims': self.ndims,
        })
        return config

    def build(self, input_shape):
        super(finiteDiffReg, self).build(input_shape)

    def call(self, x, training=None):
        if self.operator in ['lo', 'fdnu']:
            if self.ndims == 3:
                v0 = tf.cast(x, np.complex64)
                v0_f_1 = tf.signal.fft3d(v0[..., 0])
                v0_f_2 = tf.signal.fft3d(v0[..., 1])
                v0_f_3 = tf.signal.fft3d(v0[..., 2])
                v0_f_r1 = tf.cast(tf.signal.ifft3d(v0_f_1 * self.op), np.float32)
                v0_f_r2 = tf.cast(tf.signal.ifft3d(v0_f_2 * self.op), np.float32)
                v0_f_r3 = tf.cast(tf.signal.ifft3d(v0_f_3 * self.op), np.float32)
                return tf.stack([v0_f_r1, v0_f_r2, v0_f_r3], axis=-1)
            if self.ndims == 2:
                v0 = tf.cast(x, np.complex64)
                v0_f_1 = tf.signal.fft2d(v0[..., 0])
                v0_f_2 = tf.signal.fft2d(v0[..., 1])
                v0_f_r1 = tf.cast(tf.signal.ifft2d(v0_f_1 * self.op), np.float32)
                v0_f_r2 = tf.cast(tf.signal.ifft2d(v0_f_2 * self.op), np.float32)
                return tf.stack([v0_f_r1, v0_f_r2], axis=-1)
        elif self.operator == 'py':
            if self.inverse:
                adTvw = self.op_class.applyInvOp(x)
            else:
                adTvw = self.op_class.applyOp(x)
            return adTvw

    def compute_output_shape(self, input_shape):
        return input_shape


# Transpose convolution layer, upsamples by size 2.
def transpose_convolution(filters, kernel, strides, padding, name, ndims):
    upSampling = getattr(KL, 'UpSampling%dD' % ndims)
    conv = getattr(KL, 'Conv%dD' % ndims)
    up_l = upSampling(size=strides, data_format="channels_first")  # ,interpolation='bilinear')
    conv_l = conv(filters, kernel, strides=1, padding=padding, name="transpose_convolution" + name,
                  data_format="channels_first")

    def f(input):
        x = up_l(input)
        x = conv_l(x)
        return x

    return f


# Keras layers used and activation functions.
def down_conv_layer(filters, kernel, strides, padding, name, ndims):
    '''
    3D convolutional layer followed by Relu activation and 3d maxpooling.
    '''
    conv = getattr(KL, 'Conv%dD' % ndims)
    pooling = getattr(KL, 'MaxPooling%dD' % ndims)
    conv_layer = conv(filters, kernel, strides=strides, padding=padding, name="down_convolution" + name,
                      data_format="channels_first")
    act_layer = ReLU()
    pooling = pooling(2, padding="valid", data_format="channels_first")

    def f(inp):
        x = conv_layer(inp)
        x = act_layer(x)
        output = pooling(x)
        return x, output

    return f


def same_conv_layer(filters, kernel, strides, padding, name, dilation, ndims):
    """
    3D convolution layer followed by LeakuRelu activation, accepts differente dilation rates.
    """
    conv = getattr(KL, 'Conv%dD' % ndims)
    conv_layer = conv(filters, kernel, strides=strides, dilation_rate=dilation, padding=padding,
                      name="down_convolution" + name, data_format="channels_first")
    act_layer = LeakyReLU()

    def f(input):
        x = conv_layer(input)
        output = act_layer(x)
        return output

    return f


# BN disabled because we use only batch_size = 1.
def bn_conv_layer(filters, kernel, strides, padding, name, pool=True, ndims=2):
    conv = getattr(KL, 'Conv%dD' % ndims)
    pooling = getattr(KL, 'MaxPooling%dD' % ndims)
    conv_layer = conv(filters, kernel, strides=strides, padding=padding, name="bn_conv" + name,
                      data_format="channels_first")
    bn_layer = BatchNormalization(name="bn_n" + name)
    act_layer = ReLU()
    pool_layer = pooling(2, padding="valid", data_format="channels_first")

    def f(input):
        x = conv_layer(input)
        # x = bn_layer(x)
        x = act_layer(x)
        if pool:
            x = pool_layer(x)
        return x

    return f


def up_conv_layer(filters, kernel, strides, padding, shape, name, ndims):
    """
    Normal 3D convolutional layer, followed by ReLU activation and Up convolutional layer.
    """
    conv = getattr(KL, 'Conv%dD' % ndims)
    convTranspose = getattr(KL, 'Conv%dDTranspose' % ndims)

    conv_layer = conv(filters, kernel, strides=strides, padding=padding, name="scale_convolution" + name,
                      data_format="channels_first")
    act_layer = ReLU()
    deconv_layer = convTranspose(filters, 2, strides=2, padding=padding, input_shape=shape,
                                 name="up_convolution" + name, data_format="channels_first")

    def f(input):
        x = deconv_layer(input)
        # if ndims == 3:
        #     x.set_shape((None, shape[0] * strides, shape[1] * strides, shape[2] * strides, filters))
        # if ndims == 2:
        #     x.set_shape((None, shape[0] * strides, shape[1] * strides, filters))
        output = x
        return output

    return f


def fc_layer(units, name, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
    full_layer = Dense(units, name="fully_connected" + name, kernel_initializer=kernel_initializer,
                       bias_initializer='zeros', activation="sigmoid")

    def f(input):
        x = full_layer(input)
        output = x
        return output

    return f


def gauss_blur_layer(sigma, size, name):
    """
    Gaussian blur layer, aplies a guassian blur as a convolution given the gaussian parameters. Has no trainable parameters.
    """
    # Gaussian blur layer as fixed weights conv layer.
    # Generate filter.
    d1, d2, d3 = [(ss - 1.) / 2. for ss in (size, size, size)]
    y, x, z = np.ogrid[-d1:d1 + 1, -d2:d2 + 1, -d3:d3 + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2. * sigma * sigma))
    sum = np.sum(h)
    h = h / sum

    # Create layer.
    g_layer = DepthwiseConv3D(kernel_size=(size, size, size), depth_multiplier=1, use_bias=False, groups=3,
                              padding='same', name=name)

    def f(x):
        nchan = x.shape[-1]
        h2 = np.expand_dims(h, axis=-1)
        h2 = np.repeat(h2, nchan, axis=-1)
        h2 = np.expand_dims(h2, axis=-1)
        output = g_layer(x)
        g_layer.set_weights([h2])
        g_layer.trainable = False
        return output

    return f
