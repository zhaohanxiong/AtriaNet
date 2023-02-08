import os
import cv2
import sys
import h5py
import scipy.io
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from QingXia_Utils import equalize_adapthist_3d

def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)
	return

# path size
n1 = 272 # x
n2 = 272 # y

### Utah Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop/Atria_Data/AWT Utah Bi-Atria")

N_train_patients = 70 # patients to use for train set

# list all the files in training and testing sets
files = os.listdir()

# Train Data: loop through all training patients
train_Image,train_Label,train_AWT = [],[],[]
test_Image,test_Label,Barycenter,test_AWT = [],[],[],[]

for i in range(N_train_patients):

	print(str(i+1)+" Processing Train Set: "+files[i])

	# load awt file
	temp_awt = scipy.io.loadmat(files[i]+"/AWT.mat")["AWT"]

	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")
	
	# load lgemri data
	temp_img = np.zeros_like(temp_awt)
	for n in range(temp_img.shape[2]):
		
		# load image 8-bit
		temp_img[:,:,n] = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	
	# load label data one slice at a time
	for n in range(0,len(lab_files),2):
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp_lab = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# filter
		temp_lab[temp_lab > 4]  = 0 # remove septum

		# if there are positive pixels in the slice
		if np.sum(temp_lab) > 0:
			
			# find the center of mass of the mask
			midpoint = ndimage.measurements.center_of_mass(temp_lab > 0)
		
			# extract the patches from the midpoint
			n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
			n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

			# local image label for scan
			train_Image.append(temp_img[n11:n12,n21:n22,n])
			train_Label.append(temp_lab[n11:n12,n21:n22])
			train_AWT.append(temp_awt[n11:n12,n21:n22,n])

# Test Data: loop through all training patients
for i in range(N_train_patients,len(files)):

	print(str(i+1)+" Processing Test Set: "+files[i])

	# load awt file
	temp_awt = scipy.io.loadmat(files[i]+"/AWT.mat")["AWT"]
	x,y,z    = temp_awt.shape
	temp_awt = temp_awt[(x//2-288):(x//2+288),(y//2-288):(y//2+288),::2]
	
	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")
	
	# load lgemri data
	temp_img = np.zeros([x,y,z])
	for n in range(len(img_files)):
		
		# load image 8-bit
		temp_img[:,:,n] = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	temp_img = temp_img[(x//2-288):(x//2+288),(y//2-288):(y//2+288),::2]
	
	# loop through all the label slices
	patient_lab,patient_mid = np.zeros([576,576,44]),np.zeros([44,2])

	for n in range(0,len(lab_files),2):
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp_lab = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# filter 
		temp_lab[temp_lab > 4]  = 0 # remove septum

		# crop so all inputs are 576 x 576
		patient_lab[:,:,n//2] = temp_lab[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
		patient_mid[n//2,:]   = ndimage.measurements.center_of_mass(temp_lab > 0)

	# local image label for scan
	test_Image.append(temp_img)
	test_Label.append(patient_lab)
	Barycenter.append(patient_mid)
	test_AWT.append(temp_awt)
	
### Save Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("C:/Users/zxio506/Desktop")

create_folder("UtahAWT Test Set")
create_folder("UtahAWT Test Set/log")
create_folder("UtahAWT Test Set/Prediction Sample")

# Train Data
train_Image,train_Label,train_AWT = np.array(train_Image),np.array(train_Label),np.array(train_AWT)
train_AWT[train_AWT>20] = 0

# encoding label to neural network output format (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo)
temp = np.empty(shape=[train_Label.shape[0],n1,n2,5])
for i in range(5):
	x = train_Label == i
	temp[:,:,:,i] = x

train_Image,train_Label,train_AWT = np.reshape(train_Image,newshape=[-1,n1,n2,1]),np.reshape(temp,newshape=[-1,n1,n2,5]),np.reshape(train_AWT,newshape=[-1,n1,n2,1])

# Test Data
test_Image,test_Label,Barycenter,test_AWT = np.array(test_Image),np.array(test_Label),np.array(Barycenter),np.array(test_AWT)
test_AWT[test_AWT>20] = 0

# create a HDF5 dataset
print("---------- Saving Training Data")
h5f = h5py.File('UtahAWT Test Set/Training.h5','w')
h5f.create_dataset("image", data=train_Image)
h5f.create_dataset("label", data=train_Label)
h5f.create_dataset("awt",   data=train_AWT)
h5f.close()

# create a HDF5 dataset
print("---------- Saving Testing Data")
h5f = h5py.File('UtahAWT Test Set/Testing.h5','w')
h5f.create_dataset("image",    data=test_Image)
h5f.create_dataset("label",    data=test_Label)
h5f.create_dataset("centroid", data=Barycenter)
h5f.create_dataset("awt",      data=test_AWT)
h5f.close()
