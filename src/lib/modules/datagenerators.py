from ..modules.dataloader import *
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.io


def data_gen_mem(path, batch_size=1, standarize=True):
    data = load_all(path, standarize)
    length = len(data)
    data = np.concatenate(data)
    print("Succesfully loaded ", length, " volumenes of data.")
    while True:
        idx = np.asarray(range(length))
        idx = np.random.permutation(idx)
        curr = 0
        while curr + batch_size <= length:
            batch_idx = idx[curr:curr + batch_size]
            curr = curr + batch_size
            batch = data[batch_idx, ...]
            yield (batch)


def data_gen_disk(path, batch_size=1, standarize=True):
    file_names = get_filenames(path)
    length = len(file_names)
    while True:
        idx = np.asarray(range(length))
        idx = np.random.permutation(idx)
        curr = 0
        while curr + batch_size <= length:
            batch_idx = idx[curr:curr + batch_size]
            curr = curr + batch_size
            batch_names = [file_names[i] for i in batch_idx]
            tmp_list = []
            for file in batch_names:
                x = load_file(file, standarize)
                x = x[np.newaxis, ..., np.newaxis]
                tmp_list.append(x)
            batch = np.concatenate(tmp_list)
            yield (batch)


def data_gen_fix_atlas(path_files, path_atlas, batch_size=1, disk=False, z_size=32, standarize=True):
    if disk:
        gen = data_gen_disk(path_files, batch_size, standarize)
    else:
        gen = data_gen_mem(path_files, batch_size, standarize)
    atlas = load_file(path_atlas, standarize=True)
    atlas_1 = atlas[np.newaxis, ...]
    atlas_2 = atlas.reshape((1, atlas_1.shape[1] // 2, 2, atlas_1.shape[2] // 2, 2, atlas_1.shape[3] // 2, 2)).mean(
        6).mean(4).mean(2)
    atlas_3 = atlas.reshape((1, atlas_1.shape[1] // 4, 4, atlas_1.shape[2] // 4, 4, atlas_1.shape[3] // 4, 4)).mean(
        6).mean(4).mean(2)
    z_true = np.zeros((batch_size, z_size * 2))
    atlas_1 = atlas_1[..., np.newaxis]
    atlas_2 = atlas_2[..., np.newaxis]
    atlas_3 = atlas_3[..., np.newaxis]

    atlas_1 = np.repeat(atlas_1, batch_size, axis=0)
    atlas_2 = np.repeat(atlas_2, batch_size, axis=0)
    atlas_3 = np.repeat(atlas_3, batch_size, axis=0)
    y_list = [atlas_1, atlas_2, atlas_3, z_true]
    while True:
        X = next(gen)
        yield ([atlas_1, X], y_list)


def data_gen_pairs(path_files, batch_size=1, disk=False, z_size=32, standarize=True):
    if disk:
        gen = data_gen_disk(path_files, batch_size, standarize)
    else:
        gen = data_gen_mem(path_files, batch_size, standarize)

    z_true = np.zeros((batch_size, z_size * 2))
    while True:
        X = next(gen)
        Y_1 = next(gen)
        Y_2 = np.zeros((batch_size, Y_1.shape[1] // 2, Y_1.shape[2] // 2, Y_1.shape[3] // 2, 1))
        Y_3 = np.zeros((batch_size, Y_2.shape[1] // 2, Y_2.shape[2] // 2, Y_2.shape[3] // 2, 1))
        for i in range(0, batch_size):
            Y_2[i] = Y_1[i].reshape(
                (Y_1[i].shape[0] // 2, 2, Y_1[i].shape[1] // 2, 2, Y_1[i].shape[2] // 2, 2, 1)).mean(5).mean(3).mean(1)
            Y_3[i] = Y_2[i].reshape(
                (Y_2[i].shape[0] // 2, 2, Y_2[i].shape[1] // 2, 2, Y_2[i].shape[2] // 2, 2, 1)).mean(5).mean(3).mean(1)

        y_list = [Y_1, Y_2, Y_3, z_true]
        yield ([Y_1, X], y_list)


def data_gen_2d(samples, params, batch_size=1, shape=200, dataset=0):
    mov = [None] * batch_size
    tar = [None] * batch_size
    data = [None] * samples
    if dataset == 0:
        for i in range(0, samples):
            data[i] = generate_elipse(params, shape)
    elif dataset == 1:
        for i in range(0, samples, 2):
            data[i], data[i + 1] = generate_syn_seg(params, shape)
    elif dataset == 2:
        for i in range(0, samples, 2):
            data[i] = generate_half_rectangle(0.2, shape)
            data[i + 1] = generate_half_rectangle(0.8, shape)
    k = 0
    while True:
        for i in range(batch_size):
            mov[i] = data[k]
            # mov[i] = gaussian_filter(data[k], sigma=1)
            tar[i] = data[k + 1]
            # tar[i] = gaussian_filter(data[k+1], sigma=1)
            k = k + 2
            if k >= samples:
                k = 0
        yield ([np.asarray(mov), np.asarray(tar)])


def generate_elipse(params, shape=200):
    m1, d1, m2, d2 = params
    a1, b1 = np.abs(np.random.normal(m1, d1, (2)))
    a2, b2 = np.abs(np.random.normal(m2, d2, (2)))
    xx, yy = np.mgrid[:shape, :shape]
    elipse1 = ((xx - shape / 2) ** 2) / (a1 ** 2) + ((yy - shape / 2) ** 2) / (b1 ** 2)
    elipse2 = ((xx - shape / 2) ** 2) / (a2 ** 2) + ((yy - shape / 2) ** 2) / (b2 ** 2)
    donut = np.logical_and(elipse1 > 1, elipse2 < 1)
    donut = np.asarray(donut[..., np.newaxis], dtype=np.float32)
    donut = gaussian_filter(donut, sigma=1)

    return donut


def generate_syn_seg(sections=2, shape=128):
    # Generate Centers
    grid_size = shape // sections
    # centers = (np.random.random((sections ** 2, 2))/5 + 0.5) * (grid_size - 1)
    centers = (np.random.random((sections ** 2, 2))) * (grid_size - 1)
    grid = list(range(0, grid_size * (sections - 1) + 1, grid_size))
    grid = np.meshgrid(grid, grid)
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape((sections ** 2, 2))
    centers_1 = centers + grid
    centers_2 = centers_1 + (np.random.random((sections ** 2, 2))) * (grid_size - 1) * 0.4 - 0.2

    # Generate points
    grid = list(range(0, shape, 1))
    grid = np.meshgrid(grid, grid)
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape((shape, shape, 2))

    return aux_fill(sections, shape, grid, centers_1), aux_fill(sections, shape, grid, centers_2)


def aux_fill(sections, shape, grid, centers):
    dists = []
    for center in centers:
        dists += [np.sqrt(np.sum((grid - center) ** 2, axis=2))]
    dists = np.stack(dists, axis=-1)
    points = np.argmin(dists, axis=2) + 1
    mask = np.min(dists, axis=2) < 10
    # points = points * mask
    image = points[:, :, np.newaxis]
    image = np.asarray(image, dtype=np.float32)

    # image = gaussian_filter(image, sigma=1)

    # Normalize
    image = image / (sections ** 2)

    # Border conditions
    image[0:5, :] = 0
    image[:, 0:5] = 0
    image[-5:, :] = 0
    image[:, -5:] = 0
    return image


def generate_half_rectangle(cut, shape=128):
    image = np.zeros((shape, shape, 1)) + 0.5

    image[:, 0:int(shape * cut)] = 1
    # Border conditions
    image[0:5, :] = 0
    image[:, 0:5] = 0
    image[-5:, :] = 0
    image[:, -5:] = 0

    image = np.asarray(image, dtype=np.float32)
    return image


def data_gen_2d_slices(path, batch_size=1, standarize=True, random=True):
    mat = scipy.io.loadmat(path)["slices"]
    length = mat.shape[2]
    # Standarize
    max = np.max(mat, axis=(0, 1))
    min = np.min(mat, axis=(0, 1))
    data = (mat - min) / (max - min + 1E-10)
    print("Succesfully loaded ", length, " volumenes of data.")

    while True:
        idx = np.asarray(range(length))
        if random:
            idx = np.random.permutation(idx)
        curr = 0
        while curr + 2 * batch_size <= length:
            batch_idx = idx[curr:curr + 2 * batch_size]
            curr = curr + 2 * batch_size
            batch = data[..., batch_idx]
            batch = np.transpose(batch, [2, 0, 1])
            batch = batch[..., np.newaxis]
            yield ([batch[0:batch_size, ...], batch[batch_size:batch_size * 2, ...]])


def data_gen_2d_nirep(path, standarize=True):
    mat = scipy.io.loadmat(path)["slices"]
    length = mat.shape[2]
    # Standarize
    max = np.max(mat, axis=(0, 1))
    min = np.min(mat, axis=(0, 1))
    data = (mat - min) / (max - min + 1E-10)
    print("Succesfully loaded ", length, " volumenes of data.")

    while True:

        batch = data[..., 0]
        batch = batch[..., np.newaxis]
        batch = np.transpose(batch, [2, 0, 1])
        fixed = batch[..., np.newaxis]
        curr = 1
        while curr < length:
            batch = data[..., curr]
            batch = batch[..., np.newaxis]
            batch = np.transpose(batch, [2, 0, 1])
            moving = batch[..., np.newaxis]
            curr = curr + 1
            yield ([moving, fixed])
