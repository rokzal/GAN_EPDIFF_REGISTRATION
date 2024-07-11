import numpy as np
from scipy.interpolate import RegularGridInterpolator

def dice_score(phi_orig, seg_mov, seg_fix, segments=32):
    shape = phi_orig.shape
    ndims = shape[-1]
    phi = np.zeros_like(phi_orig)
    # Transform moving segmentations.
    if ndims == 2:
        X = np.linspace(0, shape[0] - 1, shape[0])
        Y = np.linspace(0, shape[1] - 1, shape[1])

        xg, yg = np.meshgrid(X, Y, indexing='ij')
        interpn = RegularGridInterpolator((X, Y), seg_mov, fill_value=0, bounds_error=False, method="nearest")

        phi[:, :, 0] = -phi_orig[:, :, 0] + xg
        phi[:, :, 1] = -phi_orig[:, :, 1] + yg
        seg_mov_def = interpn(phi)

    if ndims == 3:
        X = np.linspace(0, shape[0] - 1, shape[0])
        Y = np.linspace(0, shape[1] - 1, shape[1])
        Z = np.linspace(0, shape[2] - 1, shape[2])

        xg, yg, zg = np.meshgrid(X, Y, Z, indexing='ij')
        interpn = RegularGridInterpolator((X, Y, Z), seg_mov, fill_value=0, bounds_error=False, method="nearest")

        phi[:, :, :, 0] = -phi_orig[:, :, :, 0] + xg
        phi[:, :, :, 1] = -phi_orig[:, :, :, 1] + yg
        phi[:, :, :, 2] = -phi_orig[:, :, :, 2] + zg
        seg_mov_def = interpn(phi)

    # Calculate DSC
    inter_vol = seg_mov_def == seg_fix
    dsc = []
    for i in range(1, segments + 1):
        mov = np.sum(seg_mov_def == i)
        aux = seg_fix == i
        fix = np.sum(aux)
        inter = np.sum(inter_vol * aux)
        dsc.append(2 * inter / (mov + fix))

    return dsc

def resize_phi(phi, des_shape):
    shape = phi.shape
    factor = [0,0,0]
    factor[0] = des_shape[0] / shape[0]
    factor[1] = des_shape[1] / shape[1]
    factor[2] = des_shape[2] / shape[2]

    xl = np.linspace(0, shape[0] - 1, shape[0])
    yl = np.linspace(0, shape[1] - 1, shape[1])
    zl = np.linspace(0, shape[2] - 1, shape[2])
    X, Y, Z = np.meshgrid(xl, yl, zl, indexing='ij')
    X = X * factor[0]
    Y = Y * factor[1]
    Z = Z * factor[2]

    xl2 = np.linspace(0, des_shape[0] - 1, des_shape[0])
    yl2 = np.linspace(0, des_shape[1] - 1, des_shape[1])
    zl2 = np.linspace(0, des_shape[2] - 1, des_shape[2])
    XI, YI, ZI = np.meshgrid(xl2, yl2, zl2, indexing='ij')

    aux = np.zeros(des_shape+[3])
    phi_up = np.zeros_like(aux)
    aux[...,0] = XI
    aux[...,1] = YI
    aux[...,2] = ZI
    interpn = RegularGridInterpolator((xl * factor[0], yl * factor[1], zl * factor[2]), phi[:, :, :, 0] * factor[0] , fill_value=0, bounds_error=False)
    phi_up[..., 0] = interpn(aux)
    interpn = RegularGridInterpolator((xl * factor[0], yl * factor[1], zl * factor[2]), phi[:, :, :, 1] * factor[1], fill_value=0, bounds_error=False)
    phi_up[..., 1] = interpn(aux)
    interpn = RegularGridInterpolator((xl * factor[0], yl * factor[1], zl * factor[2]), phi[:, :, :, 2] * factor[2], fill_value=0, bounds_error=False)
    phi_up[..., 2] = interpn(aux)

    return phi_up
def jacDet_stats(warp):
    ndims = warp.shape[-1]
    shape = warp.shape
    cut = 5
    if ndims == 2:
        X = np.linspace(0, shape[0] - 1, shape[0])
        Y = np.linspace(0, shape[1] - 1, shape[1])

        xg, yg = np.meshgrid(X, Y, indexing='ij')
        warp[:, :, 0] = warp[:, :, 0] + xg
        warp[:, :, 1] = warp[:, :, 1] + yg

        grad = gradients(warp)
        J = grad[0][...,0] * grad[1][...,1] - grad[0][...,1] * grad[1][...,0]
        J = J[cut:-cut,cut:-cut]
    elif ndims ==3:
        X = np.linspace(0, shape[0] - 1, shape[0])
        Y = np.linspace(0, shape[1] - 1, shape[1])
        Z = np.linspace(0, shape[2] - 1, shape[2])

        xg, yg, zg = np.meshgrid(X, Y, Z, indexing='ij')
        warp[:, :, 0] = warp[:, :, 0] + xg
        warp[:, :, 1] = warp[:, :, 1] + yg
        warp[:, :, 2] = warp[:, :, 2] + zg

        grad = gradients(warp)
        J = grad[0][ ..., 0] * grad[1][ ..., 1] * grad[1][ ..., 1] \
            - grad[0][ ..., 0] * grad[1][ ..., 2] * grad[2][ ..., 1] \
            - grad[0][...,1] * grad[1][...,0] * grad[2][...,2] \
            + grad[0][...,1] * grad[1][...,2] * grad[2][...,0] \
            + grad[0][...,2] * grad[1][...,0] * grad[2][...,1] \
            + grad[0][ ..., 2] * grad[1][ ..., 1] * grad[2][ ..., 0]

        J = J[cut:-cut, cut:-cut, cut:-cut]
    stats = [np.max(J),np.min(J),np.sum(J<0)]

    return stats

def gradients(y):
    vol_shape = y.shape
    ndims = len(vol_shape)
    df = [None] * ndims
    for i in range(ndims-1):
        d = i
        # permute dimensions to put the ith dimension first
        r = [d, *range(d), *range(d + 1 , ndims)]
        yt = np.transpose(y, r)
        dfi = yt[2:, ...] - yt[:-2, ...]
        start = yt[1:2, ...] - yt[-1:, ...]
        end = yt[0:1, ...] - yt[-2:-1, ...]
        dfi = np.concatenate([start, dfi, end], axis=0)
        dfi = dfi / 2.0

        # permute back
        r = [*range(1, d + 1), 0, *range(d + 1, ndims)]
        df[i] = np.transpose(dfi, r)
    # Fix changed dims
    #df[0], df[1] = df[1], df[0]
    return df