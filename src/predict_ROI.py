import os
import sys
import cv2
import shutil
import argparse
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from Utils import *

# set model directory
model_path = "UtahWaikato Test Set ROI"

# get data mean and standard deviation from log
data_mean = 0.11141 #np.mean(train_data["image"])
data_std = 0.14977 #np.std(train_data["image"])

# define patch size
n1 = 272
n2 = 272

# set data directory
parser = argparse.ArgumentParser()
parser.add_argument("--path", default = "/hpc/zxio506/Atria_Data/Utah Bi-Atria/CARMA0046/pre/lgemri.nrrd", type = str)
parser.add_argument("--out_dir", default = "out", type = str)
args = parser.parse_args(sys.argv[1:])

# build and load pre-trained model
model = AtriaNet_ROI(n1, n2)
model.load(model_path+"/log/LARAmodel_UtahWaikato")

# set path
path = args.path

if ".nrrd" in path: # use this block of code to load nrrd files
	
	# load files
	img = load_nrrd(path)
	img = np.rollaxis(img,0,3)

	# create prediction with same 
	pred = np.zeros_like(img)

else: # use this block for image stacks

	# get all patient files
	files_i = np.sort(os.listdir(path))

	# read first file to get image dimension
	temp = cv2.imread(os.path.join(path,files_i[0]),cv2.IMREAD_GRAYSCALE)
	x,y = temp.shape
	z = len(files_i)

	# compile each MRI image into a stack by their centroids
	img,pred = np.zeros([x,y,z]),np.zeros([x,y,z])

	# read image
	for j in range(len(files_i)):
		img[:,:,j] = cv2.imread(os.path.join(path,files_i[j]),cv2.IMREAD_GRAYSCALE)

# preprocess
img = equalize_adapthist_3d(img / np.max(img))

# make prediction
midpoints = []
for j in range(img.shape[2]):

	# find the center of mass of the mask
	data_i = cv2.resize(img[:,:,j], (n1,n2))
	data_i = [(data_i[:,:,None] - data_mean)/data_std]
	data_sparse = np.argmax(model.predict(data_i), 3)[0]

	# compute midpoint
	midpoint = np.asarray(center_of_mass(data_sparse>0))
	
	midpoint[0] = midpoint[0]*img.shape[0]/n1
	midpoint[1] = midpoint[1]*img.shape[1]/n2
	
	# shift if too far away from centre
	midpoint[0] = min(midpoint[0], img.shape[0] - n1//2)
	midpoint[0] = max(midpoint[0], n1//2)
	midpoint[1] = min(midpoint[1], img.shape[1] - n2//2)
	midpoint[1] = max(midpoint[1], n2//2)
	
	# store
	midpoints.append(midpoint)

midpoints = np.array(midpoints)

# set up output direcotry
output_dir = args.out_dir
if os.path.isdir(output_dir):
	shutil.rmtree(output_dir)

# write to output
os.mkdir(output_dir)
output_file = "roi.npy"
np.save(os.path.join(output_dir, output_file), midpoints)
