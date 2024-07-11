import numpy as np
import tensorflow as tf
import itertools
import tensorflow.keras.backend as K

from src.ext.neurite.utils import interpn_vox
from src.ext.voxelmorph.utils import transform


def clencurt(n):
    theta = np.pi * np.asarray(range(0, n + 1)) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    ii = np.asarray(range(2 - 1, n))
    v = np.ones(n - 1)
    if n % int(2) == 0:
        w[0] = 1 / (n ** 2 - 1)
        w[n] = w[0]
        for k in range(1, int(n) // 2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
        v = v - np.cos(n * theta[ii]) / (n ** 2 - 1)
    else:
        w[0] = 1 / n ** 2;
        w[n] = w[0]
        for k in range(1, (n - 1) / 2 + 1):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
    w[ii] = 2 * v / n
    return x, w


def generate_grid(shape, unit_domain=False, dtype=np.float32):
    dims = len(shape)
    assert dims == 2 or dims == 3, "Generate grids fail, dimensions requested not 2 or 3, but " + str(dims)
    lspaces = [0] * dims

    # Generate grid
    if unit_domain:
        min = [0.0] * dims
        max = [1 - 1.0 / x for x in shape] * dims
    else:
        min = [0.0] * dims
        max = [x - 1 for x in shape]

    for i in range(0, dims):
        lspaces[i] = tf.linspace(min[i], max[i], shape[i])

    if dims == 2:
        grid = tf.meshgrid(lspaces[0], lspaces[1], indexing='ij')
        # Put channels as last dim
        grid = tf.transpose(grid, [1, 2, 0])
    elif dims == 3:
        grid = tf.meshgrid(lspaces[0], lspaces[1], lspaces[2], indexing='ij')
        # Put channels as last dim
        grid = tf.transpose(grid, [1, 2, 3, 0])

    return tf.cast(grid, dtype=dtype)


def interpn(image, grid):
    # get shapes
    shape = image.shape[:-1]
    dims = len(shape)
    max_ind = [x - 1 for x in shape]

    # Transform from unit domain to pixel coordinates.
    # Asume pixel coordinates
    # grid = grid * shape

    # Get the nearest corner points
    # grid_floor = tf.cast(tf.floor(grid), 'int32')
    grid_floor = tf.floor(grid)
    grid_ceil = grid_floor + 1

    # Clip to image boundaries.
    floors = []
    ceils = []
    for k in range(0, dims):
        floors.append(tf.clip_by_value(grid_floor[..., k], 0, max_ind[k]))
        ceils.append(tf.clip_by_value(grid_ceil[..., k], 0, max_ind[k]))

    # Separate for 2 and 3 dims .
    if dims == 2:
        # Get corner pixel values.
        Ina = tf.cast(tf.stack([floors[0], floors[1]], -1), dtype=tf.int32)
        Inb = tf.cast(tf.stack([floors[0], ceils[1]], -1), dtype=tf.int32)
        Inc = tf.cast(tf.stack([ceils[0], floors[1]], -1), dtype=tf.int32)
        Ind = tf.cast(tf.stack([ceils[0], ceils[1]], -1), dtype=tf.int32)

        # print(image.dtype)
        Ia = tf.gather_nd(image, Ina, batch_dims=0)
        Ib = tf.gather_nd(image, Inb, batch_dims=0)
        Ic = tf.gather_nd(image, Inc, batch_dims=0)
        Id = tf.gather_nd(image, Ind, batch_dims=0)

        # Calculate deltas
        wa = tf.expand_dims((ceils[0] - grid[..., 0]) * (ceils[1] - grid[..., 1]), axis=-1)
        wb = tf.expand_dims((ceils[0] - grid[..., 0]) * (grid[..., 1] - floors[1]), axis=-1)
        wc = tf.expand_dims((grid[..., 0] - floors[0]) * (ceils[1] - grid[..., 1]), axis=-1)
        wd = tf.expand_dims((grid[..., 0] - floors[0]) * (grid[..., 1] - floors[1]), axis=-1)

        image = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    elif dims == 3:
        # Get corner pixel values.
        Ina = tf.cast(tf.stack([floors[0], floors[1], floors[2]], -1), dtype=tf.int32)
        Inb = tf.cast(tf.stack([floors[0], ceils[1], floors[2]], -1), dtype=tf.int32)
        Inc = tf.cast(tf.stack([ceils[0], floors[1], floors[2]], -1), dtype=tf.int32)
        Ind = tf.cast(tf.stack([ceils[0], ceils[1], floors[2]], -1), dtype=tf.int32)

        Ine = tf.cast(tf.stack([floors[0], floors[1], ceils[2]], -1), dtype=tf.int32)
        Inf = tf.cast(tf.stack([floors[0], ceils[1], ceils[2]], -1), dtype=tf.int32)
        Ing = tf.cast(tf.stack([ceils[0], floors[1], ceils[2]], -1), dtype=tf.int32)
        Inh = tf.cast(tf.stack([ceils[0], ceils[1], ceils[2]], -1), dtype=tf.int32)

        Ia = tf.gather_nd(image, Ina, batch_dims=0)
        Ib = tf.gather_nd(image, Inb, batch_dims=0)
        Ic = tf.gather_nd(image, Inc, batch_dims=0)
        Id = tf.gather_nd(image, Ind, batch_dims=0)

        Ie = tf.gather_nd(image, Ine, batch_dims=0)
        If = tf.gather_nd(image, Inf, batch_dims=0)
        Ig = tf.gather_nd(image, Ing, batch_dims=0)
        Ih = tf.gather_nd(image, Inh, batch_dims=0)

        # Calculate deltas
        wa = tf.expand_dims((ceils[0] - grid[..., 0]) * (ceils[1] - grid[..., 1]) * (ceils[2] - grid[..., 2]), axis=-1)
        wb = tf.expand_dims((ceils[0] - grid[..., 0]) * (grid[..., 1] - floors[1]) * (ceils[2] - grid[..., 2]), axis=-1)
        wc = tf.expand_dims((grid[..., 0] - floors[0]) * (ceils[1] - grid[..., 1]) * (ceils[2] - grid[..., 2]), axis=-1)
        wd = tf.expand_dims((grid[..., 0] - floors[0]) * (grid[..., 1] - floors[1]) * (ceils[2] - grid[..., 2]),
                            axis=-1)

        we = tf.expand_dims((ceils[0] - grid[..., 0]) * (ceils[1] - grid[..., 1]) * (grid[..., 2] - floors[2]), axis=-1)
        wf = tf.expand_dims((ceils[0] - grid[..., 0]) * (grid[..., 1] - floors[1]) * (grid[..., 2] - floors[2]),
                            axis=-1)
        wg = tf.expand_dims((grid[..., 0] - floors[0]) * (ceils[1] - grid[..., 1]) * (grid[..., 2] - floors[2]),
                            axis=-1)
        wh = tf.expand_dims((grid[..., 0] - floors[0]) * (grid[..., 1] - floors[1]) * (grid[..., 2] - floors[2]),
                            axis=-1)

        image = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id, we * Ie, wf * If, wg * Ig, wh * Ih])
    return image


def ss_integrate(vel, s):
    shape_grid = vel.shape[0:-1]
    grid = generate_grid(shape_grid, dtype=vel.dtype)
    vel = vel / (2 ** s)
    # Loop
    for _ in range(s):
        vel += interpn(vel, vel + grid)

    return vel

def ss_integrate_tf(vel, s):
    shape_grid = vel.shape[0:-1]
    grid = generate_grid(shape_grid, dtype=vel.dtype)
    vel = vel / (2 ** s)
    
    # Loop
    i = tf.constant(0)
    c_v = lambda i, var: tf.less(i, s)
    b_v = lambda i, var:(i + 1, var+interpn_vox(var, var + grid))
    _, vel = tf.while_loop(c_v, b_v, [i, vel], maximum_iterations = s,back_prop=True)
    return vel

def aux_int(grid):
    def f(vel):
        vel += interpn_vox(vel, vel + grid)
        return vel

    return f


def gradientCD(y, self=None):
    vol_shape = y.get_shape().as_list()[1:-1]
    ndims = len(vol_shape)
    y = tf.cast(y, dtype=np.float)
    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        yt = K.permute_dimensions(y, r)
        dfi = yt[2:, ...] - yt[:-2, ...]
        start = yt[1:2, ...] - yt[-1:, ...]
        end = yt[0:1, ...] - yt[-2:-1, ...]
        dfi = tf.concat([start, dfi, end], axis=0)
        dfi = dfi / 2.0

        r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df[i] = tf.cast(K.permute_dimensions(dfi, r), dtype=np.float32)
        if self.unit_domain:
            df[i] = df[i] * self.shape[i]

    df[0], df[1] = df[1], df[0]
    return df


def gradientY(x, self=None):
    vol_shape = x.get_shape().as_list()[1:]
    ndims = len(vol_shape)
    d = 1

    r = [d, *range(d), *range(d + 1, ndims + 1)]

    xt = K.permute_dimensions(x, r)
    dfi = xt[2:, ...] - xt[:-2, ...]
    start = xt[1:2, ...] - xt[-1:, ...]
    end = xt[0:1, ...] - xt[-2:-1, ...]
    dfi = tf.concat([start, dfi, end], axis=0)
    dfi = dfi / 2.0

    r = [*range(1, d + 1), 0, *range(d + 1, ndims + 1)]
    dx = K.permute_dimensions(dfi, r)
    if self.unit_domain:
        dx = dx * self.shape[0]
    return tf.cast(dx, dtype=np.float32)


def gradientX(x, self=None):
    vol_shape = x.get_shape().as_list()[1:]
    ndims = len(vol_shape)
    d = 2

    r = [d, *range(d), *range(d + 1, ndims + 1)]
    xt = K.permute_dimensions(x, r)
    dfi = xt[2:, ...] - xt[:-2, ...]
    start = xt[1:2, ...] - xt[-1:, ...]
    end = xt[0:1, ...] - xt[-2:-1, ...]
    dfi = tf.concat([start, dfi, end], axis=0)
    dfi = dfi / 2.0


    r = [*range(1, d + 1), 0, *range(d + 1, ndims + 1)]
    dy = K.permute_dimensions(dfi, r)
    if self.unit_domain:
        dy = dy * self.shape[1]
    return tf.cast(dy, dtype=np.float32)


def gradientZ(x, self=None):
    vol_shape = x.get_shape().as_list()[1:]
    ndims = len(vol_shape)
    d = 3
    r = [d, *range(d), *range(d + 1, ndims + 1)]
    xt = K.permute_dimensions(x, r)
    dfi = xt[2:, ...] - xt[:-2, ...]
    start = xt[1:2, ...] - xt[-1:, ...]
    end = xt[0:1, ...] - xt[-2:-1, ...]
    dfi = tf.concat([start, dfi, end], axis=0)
    dfi = dfi / 2.0

    r = [*range(1, d + 1), 0, *range(d + 1, ndims + 1)]
    dz = K.permute_dimensions(dfi, r)
    if self.unit_domain:
        dz = dz * self.shape[2]
    return tf.cast(dz, dtype=np.float32)


def apply_op(x, op, self=None):
    x = tf.cast(x, dtype=tf.complex64)
    if self.ndims == 3:
        x_1 = tf.signal.ifft3d(tf.signal.fft3d(x[..., 0]) * op)
        x_2 = tf.signal.ifft3d(tf.signal.fft3d(x[..., 1]) * op)
        x_3 = tf.signal.ifft3d(tf.signal.fft3d(x[..., 2]) * op)
        x = tf.stack([x_1, x_2, x_3], axis=-1)

    elif self.ndims == 2:
        x_1 = tf.signal.ifft2d(tf.signal.fft2d(x[..., 0]) * op)
        x_2 = tf.signal.ifft2d(tf.signal.fft2d(x[..., 1]) * op)
        x = tf.stack([x_1, x_2], axis=-1)
    x = tf.cast(x, dtype=tf.float32)
    return x