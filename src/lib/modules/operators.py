import numpy as np


class CauchyNavierRegularizer():
	#CNFiniteDifferencesUnitDomain.
	def __init__(self,sigma = 1.0,cn_gamma=1.0,cn_alpha=None,s=1,gamma=0.0,shape = None,batch_size = 1,ndims=2):
		if ndims ==3:
			self.shape = shape
			self.batch_size = batch_size
			nx = shape[0]
			ny = shape[1]
			nz = shape[2]
			self.cn_gamma = cn_gamma
			self.cn_alpha = cn_alpha
			self.sigma = sigma
			self.s = s
			self.gamma = gamma

			sX = 2 * np.pi / nx
			sY = 2 * np.pi / ny
			sZ = 2 * np.pi / nz

			x = np.asarray(range(-nx//2,nx//2))
			y = np.asarray(range(-ny//2,ny//2))
			z = np.asarray(range(-nz//2,nz//2))

			x[0:len(x)//2] = x[0:len(x)//2] + nx
			y[0:len(y)//2] = y[0:len(y)//2] + ny
			z[0:len(z)//2] = z[0:len(z)//2] + nz

			[self.Y,self.X,self.Z] = np.meshgrid(y,x,z)

			self.Gx = -1j * np.sin(sX * self.X)*nx
			self.Gy = -1j * np.sin(sY * self.Y)*ny
			self.Gz = -1j * np.sin(sZ * self.Z)*nz

			xcoeff = 2.0 * np.cos(sX * self.X) - 2.0
			ycoeff = 2.0 * np.cos(sY * self.Y) - 2.0
			zcoeff = 2.0 * np.cos(sZ * self.Z) - 2.0

			xcoeff = xcoeff * nx * nx
			ycoeff = ycoeff * ny * ny
			zcoeff = zcoeff * nz * nz

			L = (self.cn_gamma - self.cn_alpha * (xcoeff + ycoeff + zcoeff))**self.s
			self.KK = np.fft.ifftshift(L)
		if ndims ==2:
			self.shape = shape
			self.batch_size = batch_size
			nx = shape[0]
			ny = shape[1]
			self.cn_gamma = cn_gamma
			self.cn_alpha = cn_alpha
			self.sigma = sigma
			self.s = s
			self.gamma = gamma
			L_A = np.zeros(shape)

			sX = 2 * np.pi / nx
			sY = 2 * np.pi / ny

			x = np.asarray(range(-nx//2,nx//2))
			y = np.asarray(range(-ny//2,ny//2))

			x[0:len(x)//2] = x[0:len(x)//2] + nx
			y[0:len(y)//2] = y[0:len(y)//2] + ny

			[self.Y,self.X] = np.meshgrid(y,x)

			self.Gx = -1j * np.sin(sX * self.X)*nx
			self.Gy = -1j * np.sin(sY * self.Y)*ny

			xcoeff = 2.0 * np.cos(sX * self.X) - 2.0
			ycoeff = 2.0 * np.cos(sY * self.Y) - 2.0

			xcoeff = xcoeff * nx * nx
			ycoeff = ycoeff * ny * ny

		
			L = (self.cn_gamma - self.cn_alpha * (xcoeff + ycoeff ))**self.s
			self.KK = np.fft.ifftshift(L)


