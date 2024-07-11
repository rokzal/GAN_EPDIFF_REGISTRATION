# Standar imports
import sys

# Project imports
from ..modules.tf_utils import *
from ..modules.operators import *
from ..modules.IntegrateEq_rhs import *
#from ..modules.ODE_solvers import *
# 3rd party imports
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv3D, Activation, Input, Concatenate, Conv3DTranspose

# Velocity Integration.
class FwdIntegration(Layer):
    # Custom layer, integrates time dependant velocity field.
    def __init__(self, time_steps, unit_domain=False, ndims=2, sigma=1.0, cn_gamma=1.0, cn_alpha=0.0025, s=1, gamma=0.0,
                 shape=None, batch_size=1, debug=False, operator='lo', euler=True,stationary = False,
                 **kwargs):

        assert operator in ['lo', 'py', 'fdnu']
        self.stationary = stationary
        self.operator = operator
        self.unit_domain = unit_domain
        if operator == 'lo':
            op_class = CauchyNavierRegularizer(sigma, cn_gamma, cn_alpha, s, gamma, shape=shape, ndims=ndims)
            self.L_c = tf.cast(op_class.KK,dtype=tf.complex64)
            self.K_c =  tf.cast(1.0 / op_class.KK,dtype=tf.complex64)

        elif operator == 'fdnu':
            op_class = CNFiniteDifferecnesNotUnitDomain(sigma, cn_gamma, cn_alpha, s, gamma, shape=shape)
            self.L_c = tf.cast(op_class.KK,dtype=tf.complex64)
            self.K_c = tf.cast(1.0 / op_class.KK,dtype=tf.complex64)
        # self.L_c = 1.0/op_class.KK
        # self.K_c = op_class.KK
        elif operator == 'py':
            op_class = PycaOperator(shape[0], shape[1], shape[2], 0.0025, 2, 1, cn_alpha, 1)

        self.op_class = op_class
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.shape = shape
        self.dt = 1.0 / time_steps
        self.ndims = ndims
        self.debug = debug
        self.euler = euler
        super(FwdIntegration, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FwdIntegration, self).build(input_shape)

    def compute_output_shape(self, inputShape):
        if self.debug:
            return [inputShape, inputShape, inputShape, inputShape[0:-1]]
        else:
            return inputShape

    @tf.function
    def call(self, v0, training=None):
        return self.integrate(v0)

    # @tf.custom_gradient
    def integrate(self, v0):
        disfield = tf.constant(0, shape=(self.batch_size,) + tuple(self.shape) + (self.ndims,), dtype=np.float32)
        v0 = tf.cast(v0, dtype=np.float32)

        for i in range(0, self.time_steps):
            # Disfield Integration
            w = self.jacCCFFT(disfield, v0)
            disfield = disfield - self.dt * (w + v0)
            # Velocity integration
            if not self.stationary:
                if self.euler:
                    adT_v = -self.adTranspose(v0, v0)
                    v0 = v0 + self.dt * adT_v
                else:
                    v0 = self.doRK4(v0)
        return disfield
        # return disfield, grad

    @tf.function
    def doRK4(self, v0):
        # Velocity integration
        dt = self.dt
        # v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        # k1
        k1 = self.adTranspose(v0, v0)
        # partially update v1 = v0 - (dt/6)*k1
        res = v0 - (self.dt / 6) * k1
        # k2
        arg = v0 - 0.5 * dt * k1
        k2 = self.adTranspose(arg, arg)
        # partially update v1 = v1 - (dt/3)*k2
        res = res - (dt / 3) * k2

        # k3 (stored in scratch1)
        arg = v0 - 0.5 * dt * k2
        k3 = self.adTranspose(arg, arg)
        # partially update v1 = v1 - (dt/3)*k3
        res = res - (dt / 3) * k3

        # k4 (stored in scratch2)
        arg = v0 - dt * k3
        k4 = self.adTranspose(arg, arg)
        # finish updating v1 = v1 - (dt/6)*k4
        res = res - (dt / 6) * k4

        return res

    def jacCCFFT(self, u, w):
        if self.ndims == 3:
            dx, dy, dz = gradientCD(u,   self)

            adw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1] + dz[..., 0] * w[..., 2]

            adw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1] + dz[..., 1] * w[..., 2]

            adw_3 = dx[..., 2] * w[..., 0] + dy[..., 2] * w[..., 1] + dz[..., 2] * w[..., 2]
            return tf.stack([adw_1, adw_2, adw_3], axis=-1)

        if self.ndims == 2:
            dx, dy = gradientCD(u,   self)

            adw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1]

            adw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1]

            return tf.stack([adw_1, adw_2], axis=-1)

    def adTranspose(self, v, w):
        if self.ndims == 3:
            w = tf.cast(w, dtype=np.complex64)
            # Jacobian with L operator in fourier domain.
            # 1. mft = L wft, momentum
            if self.operator == 'lo' or self.operator == 'fdnu':
                m_1 = tf.cast(tf.signal.ifft3d(tf.signal.fft3d(w[..., 0]) * self.L_c), dtype=np.float32)
                m_2 = tf.cast(tf.signal.ifft3d(tf.signal.fft3d(w[..., 1]) * self.L_c), dtype=np.float32)
                m_3 = tf.cast(tf.signal.ifft3d(tf.signal.fft3d(w[..., 2]) * self.L_c), dtype=np.float32)
            elif self.operator == 'py':
                m = self.op_class.applyOp(w)
                m_1 = m[..., 0]
                m_2 = m[..., 1]
                m_3 = m[..., 2]

            # Gradient with central differences.
            # 2. (Dft vft)T * mft
            dx, dy, dz = gradientCD(v,   self)

            adTvw_1 = dx[..., 0] * m_1 + dx[..., 1] * m_2 + dx[..., 2] * m_3
            adTvw_2 = dy[..., 0] * m_1 + dy[..., 1] * m_2 + dy[..., 2] * m_3
            adTvw_3 = dz[..., 0] * m_1 + dz[..., 1] * m_2 + dz[..., 2] * m_3

            dx = gradientX(m_1 * v[..., 0],   self)
            dy = gradientY(m_1 * v[..., 1],   self)
            dz = gradientZ(m_1 * v[..., 2],   self)

            adTvw_1 += dx + dy + dz
            
            dx = gradientX(m_2 * v[..., 0],   self)
            dy = gradientY(m_2 * v[..., 1],   self)
            dz = gradientZ(m_2 * v[..., 2],   self)

            adTvw_2 += dx + dy + dz

            dx = gradientX(m_3 * v[..., 0],   self)
            dy = gradientY(m_3 * v[..., 1],   self)
            dz = gradientZ(m_3 * v[..., 2],   self)

            adTvw_3 += dx + dy + dz

            # 5. -K adTvw_ft
            if self.operator == 'lo' or self.operator == 'fdnu':

                adTvw_1 = tf.cast(adTvw_1, dtype=np.complex64)
                adTvw_2 = tf.cast(adTvw_2, dtype=np.complex64)
                adTvw_3 = tf.cast(adTvw_3, dtype=np.complex64)

                adTvw_1 = tf.cast(tf.signal.ifft3d(self.K_c * tf.signal.fft3d(adTvw_1)), dtype=np.float32)
                adTvw_2 = tf.cast(tf.signal.ifft3d(self.K_c * tf.signal.fft3d(adTvw_2)), dtype=np.float32)
                adTvw_3 = tf.cast(tf.signal.ifft3d(self.K_c * tf.signal.fft3d(adTvw_3)), dtype=np.float32)
                adTvw = tf.stack([adTvw_1, adTvw_2, adTvw_3], axis=-1)
            elif self.operator == 'py':
                adTvw = tf.stack([adTvw_1, adTvw_2, adTvw_3], axis=-1)
                adTvw = self.op_class.applyInvOp(adTvw)
            return adTvw
        if self.ndims == 2:
            w = tf.cast(w, dtype=np.complex64)
            # Jacobian with L operator in fourier domain.
            # 1. mft = L wft, momentum
            if self.operator == 'lo' or self.operator == 'fdnu':
                m_1 = tf.cast(tf.signal.ifft2d(tf.signal.fft2d(w[..., 0]) * self.L_c), dtype=np.float32)
                m_2 = tf.cast(tf.signal.ifft2d(tf.signal.fft2d(w[..., 1]) * self.L_c), dtype=np.float32)
            elif self.operator == 'py':
                m = self.op_class.applyOp(w)
                m_1 = m[..., 0]
                m_2 = m[..., 1]

            # Gradient with central differences.
            # 2. (Dft vft)T * mft
            dx, dy = gradientCD(v,   self)

            adTvw_1 = dx[..., 0] * m_1 + dx[..., 1] * m_2
            adTvw_2 = dy[..., 0] * m_1 + dy[..., 1] * m_2

            dx = gradientX(m_1 * v[..., 0],   self)
            dy = gradientY(m_1 * v[..., 1],   self)
            adTvw_1 += dx + dy

            dx = gradientX(m_2 * v[..., 0],   self)
            dy = gradientY(m_2 * v[..., 1],   self)
            adTvw_2 += dx + dy

            # 5. -K adTvw_ft
            if self.operator == 'lo' or self.operator == 'fdnu':

                adTvw_1 = tf.cast(adTvw_1, dtype=np.complex64)
                adTvw_2 = tf.cast(adTvw_2, dtype=np.complex64)
                adTvw_1 = tf.cast(tf.signal.ifft3d(self.K_c * tf.signal.fft3d(adTvw_1)), dtype=np.float32)
                adTvw_2 = tf.cast(tf.signal.ifft3d(self.K_c * tf.signal.fft3d(adTvw_2)), dtype=np.float32)

                adTvw = tf.stack([adTvw_1, adTvw_2], axis=-1)
            elif self.operator == 'py':
                adTvw = tf.stack([adTvw_1, adTvw_2], axis=-1)
                adTvw = self.op_class.applyInvOp(adTvw)
            return adTvw