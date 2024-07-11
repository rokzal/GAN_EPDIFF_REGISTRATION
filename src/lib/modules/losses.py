# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np

from ..modules.tf_utils import *



def MSE_loss(sigma):
	def loss(y,y_pred):
		Ienergy = (y_pred - y)**2
		Ienergy = Ienergy/((sigma*sigma))
		return tf.reduce_mean(Ienergy)
	return loss

def X_MSE_loss_cust_grad(sigma, param_layer):
	@tf.custom_gradient
	def loss(y,y_pred):
		Ienergy = (y_pred - y)**2
		Ienergy = (Ienergy/(sigma*sigma))

		def grad(upstream):
			lambda_I = 2 * (y_pred - y) / (sigma * sigma)

			size = tf.cast(tf.size(y),dtype=tf.float32)
			return -lambda_I / size, lambda_I / size

		return tf.reduce_mean(Ienergy) , grad
	return loss

def NCC_loss(dim):
	win = [7] * dim
	sum_filt = tf.ones([*win, 1, 1])
	if dim ==2:
		conv_fn = getattr(tf.nn, 'conv2d')
	else :
		conv_fn = getattr(tf.nn, 'conv3d')

	sum_filt = tf.ones([*win, 1, 1])
	def loss(I,J):

		eps = 1e-5

		# compute CC squares
		I2 = I * I
		J2 = J * J
		IJ = I * J

		# compute filters

		strides = [1] * (2+dim)
		padding = 'SAME'

		# comptute local sums via convolution
		I_sum = conv_fn(I, sum_filt, strides, padding)
		J_sum = conv_fn(J, sum_filt, strides, padding)
		I2_sum = conv_fn(I2, sum_filt, strides, padding)
		J2_sum = conv_fn(J2, sum_filt, strides, padding)
		IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

		# compute cross correlation
		win_size = np.prod(win)
		u_I = I_sum / win_size
		u_J = J_sum / win_size

		cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
		I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
		J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

		cc = (cross * cross ) / (I_var * J_var + eps)

		return -tf.reduce_mean(cc)

	return loss

#We multiply before.
def energy_loss():
	def loss(_,y_pred):	
		Venergy = (y_pred)
		Venergy = tf.where(tf.math.is_nan(Venergy),tf.zeros_like(Venergy)+10,Venergy)
		return tf.reduce_mean(Venergy)*3
	return loss

def kl_loss_wrapp(k):
	ks = k
	def kl_loss_normal(y_true,y_pred):
	# Kl divergence of a distribution with diagonal covariance with respect to normal distribution with mu = 0 and sigma = 1.
	# Learnt parameter is log_sigma since we can't have negatives sigmas.

		ndims = ks
		k = tf.constant(ks,dtype='float32')
		mu = y_pred[...,0:ndims]
		log_sigma = y_pred[...,ndims:]
		kl_cost = K.sum(K.exp(log_sigma)) - K.sum(log_sigma) + K.sum(mu**2)
		res = tf.subtract(kl_cost,k)
		return 0.5 * res
	return kl_loss_normal


class X_NCC():
	'''
	Local (over window) normalized cross correlation, from : https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
	'''
	
	def __init__(self,win = [9], eps = 1e-5):
		self.win = win * 3
		self.eps = eps

	def ncc(self,I,J):
		
		conv_fn = getattr(tf.nn,'conv3d')
		
		# compute CC squares
		I2 = I*I
		J2 = J*J
		IJ = I*J

		# compute filters 
		sum_filt = tf.ones([*self.win,1,1])
		strides = [1] * 5
		padding = 'SAME'
		
		# comptute local sums via convolution
		I_sum = conv_fn(I,sum_filt,strides,padding)
		J_sum = conv_fn(J,sum_filt,strides,padding)
		I2_sum= conv_fn(I2,sum_filt,strides,padding)
		J2_sum= conv_fn(J2,sum_filt,strides,padding)
		IJ_sum= conv_fn(IJ,sum_filt,strides,padding)
		
		# compute cross correlation
		win_size = np.prod(self.win)
		u_I = I_sum/win_size
		u_J = J_sum/win_size	

		cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
		I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
		J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
		
		cc = cross*cross / (I_var*J_var + self.eps)
		
		return tf.reduce_mean(cc)
		
	def loss(self,I,J):
			return -self.ncc(I,J)

#From https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
class Grad():
    """
    N-D gradient loss, and antifolding constrain.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        return tf.reduce_mean(grad)


	
