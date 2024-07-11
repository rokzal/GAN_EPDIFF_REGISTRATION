import sys
import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling3D, Lambda, Multiply
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import src.ext.voxelmorph.layers as voxel
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer

# Project imports
from ..modules import tf_utils
from ..modules import operators
from ..modules.fwdIntegrate import FwdIntegration
from ..modules import nn_layers
from ..modules import nn_networks as nets
from ..modules.losses import *
from ..modules.model_losses import *
from ..modules.gradient_accum import *
from ..modules.metrics import *
from ..modules.callbacks import *
from ..modules.dataloader import *
from ..modules.metrics import *


class network():
    def __init__(self, sizes=[10, 10, 10], ndims=2, batch_size=1, gd_size=3, predict=False, warp_reg=False,
                 debug=False, stationary = False, cn_alpha=None, s=1, operator='lo', unit_domain=True, euler=True,
                 abla=False, mixed_prec = False, id = "",time_steps = 10, bl_size = 64):
        self.sizes = sizes
        self.statinary = stationary
        self.batch_size = batch_size
        assert ndims in [2, 3]
        self.ndims = ndims
        assert len(sizes) == ndims, "Sizes len must be equall to dimensions."
        if ndims == 3:
            assert sizes[0] % 8 == 0 and sizes[1] % 8 == 0 and sizes[
                2] % 8 == 0, "Incorrect sizes: %r , %r , %r" % sizes
        if ndims == 2:
            assert sizes[0] % 8 == 0 and sizes[1] % 8 == 0, "Incorrect sizes: %r , %r " % sizes

        self.predict = predict
        self.warp_reg = warp_reg
        self.debug = debug
        self.cn_alpha = cn_alpha
        self.s = s
        self.operator = operator
        self.unit_domain = unit_domain
        self.euler = euler
        self.abla = abla
        self.gd_size = gd_size
        self.acum_grad_gen = None
        self.acum_grad_disc_1 = None
        self.acum_grad_disc_2 = None
        self.acum_grad_disc = None
        self.cpg_manager = None
        self.cpd_manager = None
        self.cpg = None
        self.cpd = None
        self.mixed_prec = mixed_prec
        self.id = id
        self.time_steps = time_steps
        self.bl_size = bl_size

    def GAN_diff_static(self):
        shape = np.concatenate((np.asarray(self.sizes, dtype='int32'), np.asarray([1])))
        fixed = Input(shape=shape, name="Fixed_input")
        moving = Input(shape=shape, name="Moving_input")
        reference = Input(shape=shape, name="Reference_input")
        discriminator = nets.discriminator_net(redux=True, ndims=self.ndims)

        generator = nets.generator_net(self.sizes, redux=True, unit_domain=self.unit_domain, ndims=self.ndims)
        if self.ndims == 2:
            trans_cl_cf = [0, 3, 1, 2]
            trans_cf_cl = [0, 2, 3, 1]
        else:
            trans_cl_cf = [0, 4, 1, 2, 3]
            trans_cf_cl = [0, 2, 3, 4, 1]
        # vel
        fixed_t = tf.transpose(fixed, trans_cl_cf)
        moving_t = tf.transpose(moving, trans_cl_cf)
        reference_t = tf.transpose(reference, trans_cl_cf)
        vel = generator(fixed_t, moving_t)
        vel = tf.transpose(vel, trans_cf_cl)
        reg_vel = nets.finiteDiffReg(shape=self.sizes, inverse=False, batch_size=self.batch_size, s=self.s,
                                     cn_alpha=self.cn_alpha, operator=self.operator, ndims=self.ndims)(vel)


        if self.unit_domain:
            fix = [self.sizes[1], self.sizes[0]]
            if self.ndims == 3:
                fix = fix + [self.sizes[2]]
            corr_vel = Lambda(lambda x: x * fix)(vel)
        else:
            corr_vel = vel

        warp = voxel.VecInt(method='ss', name="exp_layer", int_steps=4,  indexing='xy')(corr_vel)
        warp = tf.cast(warp,dtype="float32")

        fake = voxel.SpatialTransformer(interp_method='linear', name="warped")([moving, warp])

        fake = tf.cast(fake, tf.float32)

        positive = discriminator(fixed_t, reference_t)
        disc_model = Model([fixed, reference], positive, name="Discriminator")

        if self.warp_reg:
            gen_model = Model([fixed, moving], [fake, Multiply()([vel, reg_vel])])
        else:
            gen_model = Model([fixed, moving], [fake])

        if self.predict:
            return Model([fixed, moving], [fake, warp, vel, reg_vel])
        else:
            self.gen_model = gen_model
            self.disc_model = disc_model
            self.predict_model = Model([fixed, moving], [fake, warp])
            return gen_model, disc_model

    def GAN_diff_time(self):
        shape = np.concatenate((np.asarray(self.sizes, dtype='int32'), np.asarray([1])))
        fixed = Input(shape=shape, name="Fixed_input")
        moving = Input(shape=shape, name="Moving_input")
        reference = Input(shape=shape, name="Reference_input")
        discriminator = nets.discriminator_net(redux=True, ndims=self.ndims)
        generator = nets.generator_net(self.sizes, redux=True, unit_domain=self.unit_domain, ndims=self.ndims)

        if self.ndims == 2:
            trans_cl_cf = [0, 3, 1, 2]
            trans_cf_cl = [0, 2, 3, 1]
        else:
            trans_cl_cf = [0, 4, 1, 2, 3]
            trans_cf_cl = [0, 2, 3, 4, 1]
        fixed_t = tf.transpose(fixed, trans_cl_cf)
        moving_t = tf.transpose(moving, trans_cl_cf)
        reference_t = tf.transpose(reference, trans_cl_cf)
        vel = generator(fixed_t, moving_t)
        vel = tf.transpose(vel, trans_cf_cl)

        reg_vel = nets.finiteDiffReg(inverse=False, shape=self.sizes, batch_size=self.batch_size, s=self.s,
                                     cn_alpha=self.cn_alpha, operator=self.operator, ndims=self.ndims)(vel)
        warp_l = FwdIntegration(self.time_steps, unit_domain=self.unit_domain, ndims=self.ndims, shape=self.sizes,
                                cn_alpha=self.cn_alpha, s=self.s, batch_size=self.batch_size,
                                operator=self.operator, euler=self.euler, stationary=self.statinary)
        warp = warp_l(vel)
        if self.unit_domain:
            fix = [self.sizes[1], self.sizes[0]]
            if self.ndims == 3:
                fix = fix + [self.sizes[2]]
            corr_warp = Lambda(lambda x: x * fix)(warp)
        else:
            corr_warp = warp

        fake = nn_layers.SpatialTransformer(name="warped")([moving, corr_warp])

        positive = discriminator(fixed_t, reference_t)
        disc_model = Model([fixed, reference], positive, name="Discriminator")

        if self.warp_reg:
            gen_model = Model([fixed, moving], [fake, Multiply()([vel, reg_vel])])
        else:
            gen_model = Model([fixed, moving], [fake])

        if self.predict:
            return Model([fixed, moving], [fake, corr_warp, vel, reg_vel])
        else:
            self.gen_model = gen_model
            self.disc_model = disc_model
            self.predict_model = Model([fixed, moving], [fake, corr_warp])
            return self.gen_model, self.disc_model


    def compile(self, lr, lr_disc, param, energy_weights,Sim_loss = "MSE"):
        # Create callbacks and models.
        weight_reg = 1
        sigma = 0.03
        V_loss = energy_loss()

        I_loss = MSE_loss(sigma) if Sim_loss == "MSE" else NCC_loss(self.ndims)

        Grad_loss = Grad(penalty='l2')
        bce_loss = adversarial_bce_loss

        self.generator_optimizer = tf.keras.optimizers.Adam(lr,amsgrad=True)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_disc,amsgrad=True)

        if param in ['sta', 'tde']:
            self.gen_loss = generator_loss(I_loss, bce_loss, V_loss, energy_weights)
        else:
            self.gen_loss = generator_loss(I_loss, bce_loss, Grad_loss.loss, energy_weights)

        self.disc_loss = discriminator_loss
        self.mse_metric = relative_mse_metric()

        # Checkpoint manager.
        self.cpg = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.generator_optimizer, net=self.gen_model)
        self.cpd = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.discriminator_optimizer, net=self.disc_model)
        dim_str = str(self.ndims)+"d"
        self.cpg_manager = tf.train.CheckpointManager(self.cpg, '../saved_models/'+dim_str+'/checkpoints_' + self.id + '/check_gen', max_to_keep=1)
        self.cpd_manager = tf.train.CheckpointManager(self.cpd, '../saved_models/'+dim_str+'/checkpoints_' + self.id + '/check_disc', max_to_keep=1)

    def save_checkpoint(self):
        save_path = self.cpg_manager.save()
        save_path = self.cpd_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.cpg.step), save_path))
        self.cpg.step.assign_add(1)
        self.cpd.step.assign_add(1)

    def load_checkpoint(self):
        self.cpg.restore(self.cpg_manager.latest_checkpoint)
        self.cpd.restore(self.cpd_manager.latest_checkpoint)
        print("Loaded checkpoint for step {}: ".format(int(self.cpg.step)))
        self.cpg.step.assign_add(1)
        self.cpd.step.assign_add(1)
        return int(self.cpg.step)

    @tf.function
    def train_step_wrap(self, fixed, moving, alpha, step, gen_error):
        gradients_of_generator, gradients_of_discriminator, gen_loss, disc_loss = self.train_step(fixed, moving, alpha,gen_error)

        #Acumulate gradients
        self.acum_grad_gen = accumulated_gradients(self.acum_grad_gen, gradients_of_generator, self.gd_size)
        self.acum_grad_disc = accumulated_gradients(self.acum_grad_disc, gradients_of_discriminator, self.gd_size)
		
        if step + 1 % self.gd_size:
            self.generator_optimizer.apply_gradients(zip(self.acum_grad_gen, self.gen_model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(self.acum_grad_disc, self.disc_model.trainable_variables))

            self.acum_grad_disc_1 = None
            self.acum_grad_disc_2 = None
            self.acum_grad_disc = None
            self.acum_grad_gen = None	

        return gen_loss, disc_loss

    @tf.function
    def train_step(self, fixed, moving, alpha,gen_error):
        if self.mixed_prec :
            fixed = tf.cast(fixed, "float16")
            moving = tf.cast(moving, "float16")
        with tf.GradientTape(persistent=True) as tape:
            fake, energy = self.gen_model([fixed, moving])
            reference = (1 - alpha) * fixed + alpha * moving

            real_output = self.disc_model([fixed, reference])
            fake_output = self.disc_model([fixed, fake])

            gen_loss = self.gen_loss(fake, fixed, energy, fake_output)
            gen_loss_sum = gen_loss[0] + gen_error*gen_loss[1] + gen_loss[2]
            real_loss, fake_loss = self.disc_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss
		
        gradients_of_generator = tape.gradient(gen_loss_sum, self.gen_model.trainable_variables)
        gradients_of_discriminator= tape.gradient(disc_loss, self.disc_model.trainable_variables)
        self.mse_metric.update_state([fixed, moving], fake)

        return gradients_of_generator, gradients_of_discriminator, gen_loss, disc_loss

    @tf.function
    def return_and_reset_metrics(self, batches):

        train_acc = self.mse_metric.result() / batches
        self.mse_metric.reset_states()
        return train_acc

    def log_image_2d(self, writer, fixed, moving, epoch, dsc=False):
        fake, warp = self.predict_model.predict([fixed, moving])
        log_im = []

        log_im += [fake[0, :, :]]
        log_im += [fixed[0, :, :]]
        log_im += [moving[0, :, :]]
        dices = []
        jacs_stats = []
        if dsc:
            for i in range(warp.shape[0]):
                dices.append(dice_score(warp[i, ...], moving[i, ...], fixed[i, ...], int(np.max(fixed))))

            for i in range(warp.shape[0]):
                jacs_stats.append(jacDet_stats(warp[i, ...]))


            dices = np.asarray(dices)
            jacs_stats = np.asarray(jacs_stats)
            dict = {"6_DSC": np.mean(dices), "7_max_JACD":np.mean(jacs_stats[:,0]), "8_min_JACD":np.mean(jacs_stats[:,1])
                    , "9_countNegative_JACD":np.mean(jacs_stats[:,2])}
            print(dict)
            log_dictionary(writer, dict, epoch)
        with writer.as_default():
            tf.summary.image("Fake-Fixed-Moving for epoch: " + str(epoch), log_im, step=epoch)
        # writer.add_summary(sum_image)
        # writer.flush()

    def log_image_3d(self, writer, path_ims, path_segs, epoch, dsc=False):
        ims = glob.glob(str(path_ims) + ".nii")
        segs = glob.glob(str(path_segs) + ".nii")
        ims.sort()
        segs.sort()
        moving, _ = load_image(ims[0], True)
        moving = moving[np.newaxis, ..., np.newaxis]
        moving_seg, _ = load_image(segs[0], False)
        moving_seg = moving_seg[np.newaxis, ..., np.newaxis]
        dices = []

        if dsc:
            start = timer()
            for i in range(1, 16):
                fixed, _ = load_image(ims[i], True)
                fixed = fixed[np.newaxis, ..., np.newaxis]
                fixed_seg, _ = load_image(segs[i], False)
                fixed_seg = fixed_seg[np.newaxis, ..., np.newaxis]
                fake, warp = self.predict_model([fixed, moving])
                res_phi = resize_phi(np.squeeze(warp), list(fixed_seg.shape[1:4]))
                dices.append(dice_score(res_phi, np.squeeze(moving_seg), np.squeeze(fixed_seg), segments=32))
            dices = np.asarray(dices)
            dict = {"6_DSC": np.mean(dices)}
            print(dict)
            log_dictionary(writer, dict, epoch)
            end = timer()
            print("Time for DSC Calculation = " + str((end - start) // 60) + " minutes" + str(
                (end - start) % 60) + " seconds")

        moving, _ = load_image(ims[0], True)
        moving = moving[np.newaxis, ..., np.newaxis]
        fixed, _ = load_image(ims[1], True)
        fixed = fixed[np.newaxis, ..., np.newaxis]
        fake, warp = self.predict_model([fixed, moving])
        log_im = []

        size_cut = fake.shape[3] // 2
        log_im += [fake[0, :, size_cut, :]]
        log_im += [fixed[0, :, size_cut, :]]
        log_im += [moving[0, :, size_cut, :]]

        with writer.as_default():
            tf.summary.image("Fake-Fixed-Moving for epoch: " + str(epoch), log_im, step=epoch)
