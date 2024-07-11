from argparse import ArgumentParser
import os
import datetime
import time
import sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

sys.path.append("..")
from lib.modules.nn_models import *
from lib.modules.datagenerators import data_gen_fix_atlas
from lib.modules.dataloader import *

def register(moving, fixed, results_path, model_path, standarize, cn_alpha, s, model, operator):
    gpu = '/gpu:%d' % 0

    num_reg = len(moving)

    sample_number, shape = get_info(moving[0])
    #Check memory usage.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if operator in ['lo', 'py']:
        unit_domain = True
    else:
        unit_domain = False

    with tf.device(gpu):
        stationary = True if model == 'sta' else False
        cls = network(sizes=shape, debug=False, predict=True, cn_alpha=cn_alpha, operator=operator,
                      unit_domain=unit_domain, ndims=3, s=s, stationary= stationary)
        if model == 'sta':
            model = cls.GAN_diff_static()
        if model == 'tde' :
            model = cls.GAN_diff_time()

    model.load_weights(model_path, by_name=True)


    for i in range(0,num_reg):
        moving_im, affine = load_image(moving[i], standarize)
        moving_im = moving_im[np.newaxis, ..., np.newaxis]
        fixed_im, affine = load_image(fixed[i], standarize)
        fixed_im = fixed_im[np.newaxis, ..., np.newaxis]

        name = moving[i].split('/')[-1]
        name = name.split('.')[0]

        start = time.clock()
        results = model.predict([fixed_im, moving_im])
        end = time.clock()
        print("Time image " + str(i) + " :" + str(end - start))
        print("file : " + name)
        i += 1
        result = results[0]
        result_image = result[0, ..., 0]
        save_file(result_image, affine, results_path + "warped_" + name + ".nii")

        warp = results[1]
        warp = warp[0, ...]
        save_warp(warp, results_path + "disfield_" + name + ".mat")

        if len(results) >= 3:
            vel = results[2]
            vel = vel[0, ...]
            save_warp(vel, results_path + "velzero_" + name + ".mat")

    memory = tf.config.experimental.get_memory_info('GPU:0')['peak']
    print("Peak memory usage is " + str(memory / (2 ** 30)) + " GB")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--moving", dest="moving", type=str, nargs='+',
                    help="Paths to moving images.")

    parser.add_argument("--fixed", dest="fixed", type=str, nargs='+',
                        help="Paths to fixed images")

    parser.add_argument("--results_path", dest="results_path", type=str,
                        help="Directory in which to save registered images.")

    parser.add_argument("--model_path", dest="model_path", type=str,
                        help="Path of the trained model.")

    parser.add_argument("--parameterization", dest="model", type=str,
                        help="Defines network parameterization,(stationary velocity,EPDiff-constrained): ['sta','tde'].",
                        default='dis')

    parser.add_argument("--operator", dest="operator", type=str, help="Regularization operator : ['lo','py'].",
                        default='lo')

    parser.add_argument("--no_standarize", dest='standarize', default=True, action='store_false')

    parser.add_argument("--cn_alpha", dest="cn_alpha", type=float, default=0.0025, help="L regularizator cn_alpha.")

    parser.add_argument("--s", dest="s", type=float, default=1, help="L regularizator s.")

    args = parser.parse_args()

    # Disable tf warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    register(**vars(args))

    print('Registration completed.')
