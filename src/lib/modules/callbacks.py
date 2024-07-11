import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def log_dictionary(writer,dictionary,epoch):
    for item in dictionary.items():
        with writer.as_default():
            tf.summary.scalar(item[0], item[1], step=epoch)

class LogImage(tf.keras.callbacks.Callback):

    def __init__(self, image, atlas, log_dir):
        self.log_dir = log_dir
        self.image = image
        self.atlas = atlas
        self.file_writer = tf.summary.FileWriter(log_dir + "/images")
        super(LogImage, self).__init__()

    def __del__(self):
        self.file_writer.close()

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict([self.atlas, self.image])
        pred = pred[0][0]
        write_pred = pred[np.newaxis, :, pred.shape[1] // 2, ...]
        ses = K.get_session()
        with tf.Session() as ses:
            sum_image = tf.summary.image("Prediction for epoch: " + str(epoch), write_pred)
            p = ses.run(sum_image)
            self.file_writer.add_summary(p)
            self.file_writer.flush()


# Learning Rate decay function.
def learning_decay(ini_lr):
    # initial rate 1e-3
    initial_lrate = ini_lr

    def f(epoch):
        # drop = 0.3
        drop = 0.3
        # epochs_drop = 10.0
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, ((epoch) / epochs_drop))
        return lrate

    return f


class loss_weight_update(tf.keras.callbacks.Callback):
    def __init__(old, new, change):
        self.change = change
        self.old = old
        self.new = new

    def on_epoch_end(self, epoch, logs={}):
        if epoch == self.change:
            self.old = self.new
