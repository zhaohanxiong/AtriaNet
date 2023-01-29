import os
import sys
import cv2
import argparse
import numpy as np
from Utils import *

# set model directory
model_path = "UtahWaikato Test Set"

# do we calculate fibrosis
calculate_fibrosis = False
LA_thres, RA_thres = 3.5, 4.5

# get data mean and standard deviation from log
data_mean = 0.1878 #np.mean(train_data["image"])
data_std = 0.17642 #np.std(train_data["image"])

# define patch size
n1 = 272
n2 = 272

# set data directory
parser = argparse.ArgumentParser()
parser.add_argument("--path", default = "/hpc/zxio506/Atria_Data/Utah Bi-Atria/CARMA0046/pre/lgemri.nrrd", type = str)
parser.add_argument("--out_dir", default = "out", type = str)
args = parser.parse_args(sys.argv[1:])

# build and load pre-trained model
model = AtriaNet_Seg(n1, n2)
model.load(model_path+"/log/LARAmodel_UtahWaikato")

# set up output direcotry from ROI detection
output_dir = args.out_dir

# load midpoints
midpoints = np.load(os.path.join(output_dir, "roi.npy"))

# set path
path = args.path

if ".nrrd" in path: # use this block of code to load nrrd files
	
	# load files
	img = load_nrrd(path)
	img = np.rollaxis(img,0,3)

	# store original mri image for later
	img_original = np.copy(img)

	# create prediction with same 
	pred = np.zeros_like(img)

else: # use this block for image sequences

	# get all patient files
	files_i = np.sort(os.listdir(path))

	# read first file to get image dimension
	temp = cv2.imread(os.path.join(path,files_i[0]),cv2.IMREAD_GRAYSCALE)
	x,y = temp.shape
	z = len(files_i)

	# compile each MRI image into a stack
	img,pred = np.zeros([x,y,z]),np.zeros([x,y,z])

	# read image
	for j in range(len(files_i)):
		img[:,:,j] = cv2.imread(os.path.join(path,files_i[j]),cv2.IMREAD_GRAYSCALE)
	
	# store original mri image for later
	img_original = np.copy(img)

# preprocess
img = equalize_adapthist_3d(img_original / np.max(img_original))

# make prediction
for j in range(img.shape[2]):

	midpoint = midpoints[j]

	if not np.any(np.isnan(midpoint)):

		# get ranges for the images subcrop based on midpoint
		n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
		n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

		# make prediction for slices with midpoints
		data_i = [(img[n11:n12,n21:n22,j][:,:,None] - data_mean)/data_std]
		data_o = model.predict(data_i)

		pred[n11:n12,n21:n22,j] = np.argmax(data_o,3)[0]

# calculate fibrosis
if calculate_fibrosis:

	# seperate LA and RA walls
	pred_RA_dil = np.zeros_like(img)

	for i in range(img.shape[2]):
		pred_RA_dil[:,:,i] = cv2.dilate(np.uint8(pred[:,:,i]==2),np.ones((3,3),np.uint8),iterations=7)

	# get RA wall and LA wall individually
	pred_RA_wall,pred_LA_wall = np.zeros_like(img),np.zeros_like(img)
	pred_RA_wall[np.logical_and(pred==1,pred_RA_dil==1)]  = 1
	pred_LA_wall[np.logical_and(pred==1,pred_RA_wall!=1)] = 1

	# seperate fibrosis into left and right (4 = RA fibrosis, 5 = LA fibrosis)
	fib_pred_RA = utah_threshold(np.copy(img), pred_RA_wall, RA_thres)
	fib_pred_LA = utah_threshold(np.copy(img), pred_LA_wall, LA_thres)

	# add fibrosis to label
	pred[fib_pred_RA==1] = 4
	pred[fib_pred_LA==1] = 5

	# print outputs
	print("\n\n----------------------- Fibrosis")
	print("LA Fibrosis Percentage:" + str(np.sum(pred==5)/np.sum(pred_LA_wall)*100))
	print("RA Fibrosis Percentage:" + str(np.sum(pred==4)/np.sum(pred_RA_wall)*100))
	print("\n\n----------------------- Finished")

# write to output
os.mkdir(os.path.join(output_dir,"image"))
os.mkdir(os.path.join(output_dir,"prediction"))
os.mkdir(os.path.join(output_dir,"prediction_view"))

assert np.all(img.shape == pred.shape)

for i in range(pred.shape[2]):
	cv2.imwrite(os.path.join(output_dir,"image","{0:03}".format(i+1)+".png"),img_original[:,:,i])
	cv2.imwrite(os.path.join(output_dir,"prediction","{0:03}".format(i+1)+".png"),pred[:,:,i])
	cv2.imwrite(os.path.join(output_dir,"prediction_view","{0:03}".format(i+1)+".png"),pred[:,:,i]*50)
