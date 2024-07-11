import numpy as np
import tensorflow as tf
from ..modules.tf_utils import *


# Equations
def EPDiffEquation3d_rhs(var, const, ind_t, param_class):
    # Extract variables
    vt = var

    # Calculate rhs
    rhs = -adTranspose(param_class, vt, vt)

    return rhs


# Equations
def UEPDiffEquation3d_rhs(var, const, ind_t, param_class):
    # Extract variables
    U = var
    v = const[0]
    vt = v[int(ind_t)]

    # Calculate rhs
    rhs = -adTranspose(param_class, vt, U)

    return rhs


def AdjointEPDiffEquation3d_rhs(var, const, ind_t,  param_class):
    v_ = var
    v = const[0]
    U = const[1]

    vt = v[int(ind_t)]
    Ut = U[int(ind_t)]

    rhs = Ut - adTranspose(param_class, v_, vt) + ad(param_class, vt, v_)

    return rhs


def DeformationEquation3d_rhs(var, const, ind_t,  param_class):
    # Extract variables
    phi = var
    v = const[0]
    vt = v[int(ind_t)]

    # Calculate rhs
    w = jacCCFFT(param_class, phi, vt)
    rhs = -w + vt
    # rhs = -(w + vt)

    return rhs

def DeformationEquation3d_SL_rhs(var, const, param_class):
    # Extract variables
    phi = var
    v = const[0]
    vt = v[int(ind_t)]

    # Calculate rhs
    w = jacCCFFT(param_class, phi, vt)
    rhs = -w + vt
    # rhs = -(w + vt)

    return rhs


# Aux functions
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
        dx, dy, dz = gradientCD(v, self.gradient_fourier, self)

        adTvw_1 = dx[..., 0] * m_1 + dx[..., 1] * m_2 + dx[..., 2] * m_3
        adTvw_2 = dy[..., 0] * m_1 + dy[..., 1] * m_2 + dy[..., 2] * m_3
        adTvw_3 = dz[..., 0] * m_1 + dz[..., 1] * m_2 + dz[..., 2] * m_3

        dx = gradientX(m_1 * v[..., 0], self.gradient_fourier, self)
        dy = gradientY(m_1 * v[..., 1], self.gradient_fourier, self)
        dz = gradientZ(m_1 * v[..., 2], self.gradient_fourier, self)

        adTvw_1 += dx + dy + dz

        dx = gradientX(m_2 * v[..., 0], self.gradient_fourier, self)
        dy = gradientY(m_2 * v[..., 1], self.gradient_fourier, self)
        dz = gradientZ(m_2 * v[..., 2], self.gradient_fourier, self)

        adTvw_2 += dx + dy + dz

        dx = gradientX(m_3 * v[..., 0], self.gradient_fourier, self)
        dy = gradientY(m_3 * v[..., 1], self.gradient_fourier, self)
        dz = gradientZ(m_3 * v[..., 2], self.gradient_fourier, self)

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
        dx, dy = gradientCD(v, self.gradient_fourier, self)

        adTvw_1 = dx[..., 0] * m_1 + dx[..., 1] * m_2
        adTvw_2 = dy[..., 0] * m_1 + dy[..., 1] * m_2

        dx = gradientX(m_1 * v[..., 0], self.gradient_fourier, self)
        dy = gradientY(m_1 * v[..., 1], self.gradient_fourier, self)
        adTvw_1 += dx + dy

        dx = gradientX(m_2 * v[..., 0], self.gradient_fourier, self)
        dy = gradientY(m_2 * v[..., 1], self.gradient_fourier, self)
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


def ad(self, v, w):
    if self.ndims == 3:
        # 1. Dft vft * wft
        dx, dy, dz = gradientCD(v, self.gradient_fourier, self)

        advw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1] + dz[..., 0] * w[..., 2]
        advw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1] + dz[..., 1] * w[..., 2]
        advw_3 = dx[..., 2] * w[..., 0] + dy[..., 2] * w[..., 1] + dz[..., 2] * w[..., 2]

        # 2. Dft wft * vft

        dx, dy, dz = gradientCD(w, self.gradient_fourier, self)

        advw_1 += - dx[..., 0] * v[..., 0] - dy[..., 0] * v[..., 1] - dz[..., 0] * v[..., 2]
        advw_2 += - dx[..., 1] * v[..., 0] - dy[..., 1] * v[..., 1] - dz[..., 1] * v[..., 2]
        advw_3 += - dx[..., 2] * v[..., 0] - dy[..., 2] * v[..., 1] - dz[..., 2] * v[..., 2]

        advw = tf.stack([advw_1, advw_2, advw_3], axis=-1)

    elif self.ndims == 2:
        # 1. Dft vft * wft
        dx, dy = gradientCD(v, self.gradient_fourier, self)

        advw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1]
        advw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1]

        # 2. Dft wft * vft

        dx, dy = gradientCD(w, self.gradient_fourier, self)

        advw_1 += - dx[..., 0] * v[..., 0] - dy[..., 0] * v[..., 1]
        advw_2 += - dx[..., 1] * v[..., 0] - dy[..., 1] * v[..., 1]

        advw = tf.stack([advw_1, advw_2], axis=-1)

    return advw


def jacCCFFT(self, u, w):
    if self.ndims == 3:
        dx, dy, dz = gradientCD(u, self.gradient_fourier, self)

        adw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1] + dz[..., 0] * w[..., 2]

        adw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1] + dz[..., 1] * w[..., 2]

        adw_3 = dx[..., 2] * w[..., 0] + dy[..., 2] * w[..., 1] + dz[..., 2] * w[..., 2]
        return tf.stack([adw_1, adw_2, adw_3], axis=-1)

    if self.ndims == 2:
        dx, dy = gradientCD(u, self.gradient_fourier, self)

        adw_1 = dx[..., 0] * w[..., 0] + dy[..., 0] * w[..., 1]

        adw_2 = dx[..., 1] * w[..., 0] + dy[..., 1] * w[..., 1]

        return tf.stack([adw_1, adw_2], axis=-1)
