from argparse import ArgumentParser
import os
import datetime
import sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import scipy.io
from tensorflow.keras.utils import Progbar

sys.path.append("..")
from lib.modules.nn_models import *
from lib.modules.datagenerators import *
from lib.modules.dataloader import *
from lib.modules.callbacks import *


def train(data_path, model_path, logs_path, lr, s, cn_alpha, batch_size, gen_mode, epochs, verbose, energy, gpu, model,
          operator, reg, standarize, gd_size, cont_check, dsc, id, SimLoss):
    # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    # Check memory usage.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    sample_number, shape = get_info(data_path)

    if operator in ['lo', 'py']:
        unit_domain = True
    else:
        unit_domain = False
    with tf.device(gpu):
        stationary = True if model == 'sta' else False
        cls = network(batch_size=batch_size, gd_size=gd_size, sizes=shape, warp_reg=True, debug=False,
                      cn_alpha=cn_alpha, unit_domain=True, s=s, ndims=3, stationary= stationary, id = id)

        if model == 'dis' :
            gen_model, disc_model = cls.GAN_disp()
        if model == 'sta':
            gen_model, disc_model = cls.GAN_diff_static()
        if model == 'tde' :
            gen_model, disc_model = cls.GAN_diff_time()


    energy_weights = [1, 1, reg] 
    lr_disc = lr / 5
    cls.compile(lr, lr_disc, model, energy_weights,SimLoss)
    base_epoch = 0
    if cont_check:
        base_epoch = cls.load_checkpoint() - 1

    if gen_mode:
        gen = data_gen_disk(data_path, batch_size, standarize)
    else:
        gen = data_gen_mem(data_path, batch_size, standarize)

    alpha = 0.2
    best_rmse = 10000
    steps = sample_number // (batch_size * 2)

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if cont_check:
        logs = glob.glob(logs_path + str(id) + '_[0-9]*[0-9]')
        logs.sort()
        time_str_id = logs[-1].split("/")[-1]
        log_sub_dir = logs_path + time_str_id
    else:
        log_sub_dir = logs_path + id + "_" + time_str

    save_model_base = model_path + id + "_" + "3d_" + str(reg) + "_" + model + time_str
    save_model_best = save_model_base + "_best.h5"
    save_model_last = save_model_base + "_last.h5"

    summary_writer = tf.summary.create_file_writer(log_sub_dir)

    fixed_log = np.asarray(next(gen), dtype=np.float32)
    moving_log = np.asarray(next(gen), dtype=np.float32)
    for e in range(0, epochs):
        e += base_epoch
        print("Start epoch: ", str(e))
        epoch_gen_loss = []
        epoch_disc_loss = []
        progress_bar = Progbar(target=steps)

        for k in range(0, steps):
            # get data.
            fixed = np.asarray(next(gen), dtype=np.float32)
            moving = np.asarray(next(gen), dtype=np.float32)

            # Train Step
            gen_error = 1 if e >=10 else 0 
            gen_loss, disc_loss = cls.train_step_wrap(fixed, moving, alpha, steps,gen_error)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)

            progress_bar.update(k + 1)

        # At end epoch log and show progress.
        epoch_gen_loss = np.asarray(epoch_gen_loss)
        epoch_disc_loss = np.asarray(epoch_disc_loss)

        Sim_loss = np.mean(epoch_gen_loss[:, 0])
        Adv_loss = np.mean(epoch_gen_loss[:, 1])
        Reg_loss = np.mean(epoch_gen_loss[:, 2])
        Disc_loss = np.mean(epoch_disc_loss)
        rmse = cls.return_and_reset_metrics(steps)

        # Write epoch end metrics.
        if energy:
            dict = {"1_Gen_Loss": Adv_loss, "2_Disc_loss": Disc_loss, "3_I_LOSS": Sim_loss, "4_V_LOSS": Reg_loss,
                    "5_RMSE": float(rmse)}
        else:
            dict = {"1_Gen_Loss": Adv_loss, "2_Disc_loss": Disc_loss, "3_I_LOSS": Sim_loss, "4_MSE": rmse}
        print(dict)

        # cls.log_image_3d(summary_writer, dice_ims, dice_segs, e, dsc=dsc)
        # log_dictionary(summary_writer, dict, e)

        # checkpoint
        cls.save_checkpoint()

        # #Save best model
        if best_rmse > rmse:
            gen_model.save_weights(save_model_best)
            best_rmse = rmse

    gen_model.save_weights(save_model_last)
    memory = tf.config.experimental.get_memory_info('GPU:0')['peak']
    print("Peak memory usage is " + str(memory / (2 ** 30)) + " GB")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", dest="data_path", type=str,
                        help="Directory containing image files used for training.",
                        default="Please specify data path")

    parser.add_argument("--model_path", dest="model_path", type=str,
                        help="Name of the directory in wich to save the trained model",
                        default="../saved_models/")

    parser.add_argument("--logs_path", dest="logs_path", type=str,
                        help="Name of the directory in which to save TensorBoard logs.", default="../tensorboard_3d/")


    parser.add_argument("--gpu", dest="gpu", type=str, help="GPU identifier.", default="/GPU:0")

    parser.add_argument("--parameterization", dest="model", type=str,
                        help="Defines network output,(displacement,static,time dep): ['sta','tde'].",
                        default='tde')

    parser.add_argument("--operator", dest="operator", type=str, help="Regularization operator : ['lo','py','fdnu'].",
                        default='lo')

    parser.add_argument("--Sim_Loss", dest="SimLoss", type=str, help="Image similarity Loss: ['MSE','NCC'].",
                        default='MSE')

    parser.add_argument("--id", dest="id", type=str, help="Experiment ID (string).",
                        default='')

    parser.add_argument("--lr", dest="lr", type=float, default=1E-4, help="Adam optimizer learning rate.")

    parser.add_argument("--cn_alpha", dest="cn_alpha", type=float, default=0.0025, help="L regularizator cn_alpha.")

    parser.add_argument("--use_disk", dest='gen_mode', default=False, action='store_true')

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1, help="Batch size (in GPU).")

    parser.add_argument("--gd_size", dest="gd_size", type=int, default=3, help="gd_size")

    parser.add_argument("--epochs", dest='epochs', type=int, default=50,
                        help="Number of epochs for which to train the network.")

    parser.add_argument("--verbose", dest='verbose', type=int, default=1,
                        help="Whether to perform training in verbose mode or not.")

    parser.add_argument("--energy_loss", dest='energy', default=False, action='store_true')

    parser.add_argument("--reg_weight", dest='reg', type=float, default=100, help="Energy regularization weight.")

    parser.add_argument("--DSC", dest='dsc', default=False, help="Calculate DSC during training.")

    parser.add_argument("--no_standarize", dest='standarize', default=True, action='store_false')

    parser.add_argument("--s", dest='s', type=int, default=2, help="s regularization parameter.")

    parser.add_argument("--cont_checkpoint", dest='cont_check', default=False, action='store_true')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    start_time = datetime.datetime.now()

    train(**vars(args))

    time_elapsed = datetime.datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
