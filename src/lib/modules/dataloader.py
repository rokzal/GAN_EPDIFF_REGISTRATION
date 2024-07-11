import nibabel as nib
import numpy as np
import glob
import scipy.io as sio

def get_filenames(path):
	text = glob.glob(path+'*.nii')
	if len(text) > 0:
		return text
	else :
		return glob.glob(path)

def get_sample_number(path):
	list = get_filenames(path)
	return len(list)

def load_file(path,standarize = False):
	x = nib.load(path).get_fdata()
	if standarize :
		x = standarize_sample(x)
	return x

def load_image(path,standarize = False):
	x = nib.load(path)
	image = x.get_fdata()
	if standarize :
		image = standarize_sample(image)
	return image,x.affine

def save_file(array,affine,path):
	img = nib.Nifti1Image(array,affine)
	nib.save(img,path)

def save_warp(array,path):
	sio.savemat(path,{'disfield':array})

def get_shape(path):
	image = load_file(path)
	return image.shape

def get_info(path):
	list = get_filenames(path)
	return get_sample_number(path),get_shape(list[0])

def load_all(path,standarize = False):
	list = get_filenames(path)
	data = []
	for file in list:
		x = load_file(file,standarize)
		x = x[np.newaxis,...,np.newaxis]
		data.append(x)
	return data

def load_one(path,standarize = False):
	list = get_filenames(path)
	x = load_file(list[0],standarize)
	x = x[np.newaxis,...,np.newaxis]
	return x

def standarize_sample(sample):
	'''
	mean = np.mean(sample,axis=None)
	std = np.std(sample,axis=None)
	data= np.asarray(( sample - mean) / (std + 1E-10))
	'''
	max = np.max(sample,axis=None)
	min = np.min(sample,axis=None)
	data = np.asarray((sample-min)/(max-min+1E-10),dtype=np.float32)
	return data
		
