import tensorflow as tf
from ..modules import losses
from tensorflow.keras import layers


@tf.function
def adversarial_bce_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def generator_loss(im_similarity_loss, adversarial_loss, regularization_loss, weights):
    def loss(gen_image, target_image, energy, disc_fake_output):
        return [weights[0] * im_similarity_loss(gen_image, target_image), weights[1] * adversarial_loss(
            disc_fake_output), weights[2] * regularization_loss(0, energy)]

    return loss

def LDDMM_loss(im_similarity_loss, regularization_loss, weights):
    def loss(gen_image, target_image, energy):
        return [weights[0] * im_similarity_loss(gen_image, target_image), weights[1] , weights[1] * regularization_loss(0, energy)]
    return loss

@tf.function
def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_loss = real_loss + fake_loss
    return real_loss, fake_loss


class relative_mse_metric(tf.keras.metrics.Metric):

    def __init__(self, name='RMSE', **kwargs):
        super(relative_mse_metric, self).__init__(name=name, **kwargs)
        self.metric = self.add_weight(name='RMSE', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        fixed, moving = y_true
        fixed = tf.cast(fixed,"float32")
        moving = tf.cast(moving, "float32")
        y_pred = tf.cast(y_pred, "float32")
        values = tf.reduce_mean((y_pred - fixed) ** 2) / tf.reduce_mean((moving - fixed) ** 2)

        self.metric.assign_add(values)

    def result(self):
        return self.metric

    def reset_states(self):
        self.metric.assign(0)

