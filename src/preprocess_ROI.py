import sys
import os
import numpy as np
import cv2
import SimpleITK as sitk
import h5py
from scipy import ndimage
from Utils import equalize_adapthist_3d

### Helper functions
def load_nrrd(full_path_filename):
	# this function loads .nrrd files into a 3D matrix and outputs it
	# the input is the specified file path
	# the output is the N x A x B for N slices of sized A x B
	# after rolling, the output is the A x B x N
	data = sitk.ReadImage(full_path_filename)						# read in image
	data = sitk.Cast(sitk.RescaleIntensity(data),sitk.sitkUInt8)	# convert to 8 bit (0-255)
	data = sitk.GetArrayFromImage(data)								# convert to numpy array
	data = np.rollaxis(data,0,3)
	return(data)

def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)
	return

train_Image,train_Label = [],[]
test_Image,test_Label,Barycenter = [],[],[]

### Waikato Initialization -------------------------------------------------------------------------------------------------------------------------------------------
N_train_patients = [0,6,7,10] # patients to use for train set
N_test_patients  = [1,2,3,4,5,8,9] # patients to use for test set

os.chdir("/hpc/zxio506/Atria_Data/Waikato")

# list all the files in training and testing sets
files = os.listdir()

# sort files
files = np.array(files)
files = files[np.argsort([int(f.replace("v2","")) for f in files])]

### Waikato Data ----------------------------------------------------------------------------------------------------------------------------------------------
# Train: loop through all training patients
for i in N_train_patients:

	print(str(i+1)+" Processing Train Set: "+files[i])
	
	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")
	
	# load data, all must be 640 x 640 x 44
	temp_img,temp_lab = np.zeros([640,640,44]),np.zeros([640,640,44])
	for n in range(temp_img.shape[2]):
	
		# load image 8-bit
		temp_img[:,:,n] = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp_lab[:,:,n] = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)

	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	
	# filter
	temp_lab[temp_lab > 0] = 1

	for n in range(len(lab_files)):

		# crop to utah size
		x,y,_ = temp_img.shape
		temp_img_temp = temp_img[(x//2-288):(x//2+288),(y//2-288):(y//2+288),n]
		temp_lab_temp = temp_lab[(x//2-288):(x//2+288),(y//2-288):(y//2+288),n]

		# resize
		temp_img_temp = cv2.resize(temp_img_temp, (272, 272))
		temp_lab_temp = cv2.resize(temp_lab_temp, (272, 272))

		# local image label for scan
		train_Image.append(temp_img_temp)
		train_Label.append(temp_lab_temp)

# Test: loop through all training patients
for i in N_test_patients:

	print(str(i+1)+" Processing Test Set: "+files[i])
	
	# list all files in lgemri and labels
	img_files,lab_files = os.listdir(files[i]+"/lgemri"),os.listdir(files[i]+"/label")

	# load data, all must be 640 x 640 x 44
	temp_img,temp_lab,temp_mid = np.zeros([272,272,44]),np.zeros([272,272,44]),np.zeros([44,2])
	for n in range(temp_img.shape[2]):
	
		# load image 8-bit
		temp1 = cv2.imread(os.path.join(files[i]+"/lgemri",img_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
		temp2 = cv2.imread(os.path.join(files[i]+"/label",lab_files[n]),cv2.IMREAD_GRAYSCALE)
		
		# crop to utah size
		x,y = temp1.shape
		temp_img_temp = temp1[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
		temp_lab_temp = temp2[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
		
		# resize
		temp_img[:,:,n] = cv2.resize(temp_img_temp, (272, 272))
		temp_lab[:,:,n] = cv2.resize(temp_lab_temp, (272, 272))
		
		# find the center of mass of the mask
		temp_mid[n,:] = ndimage.measurements.center_of_mass(temp_lab[:,:,n] > 0)

	# normalize data
	temp_img = equalize_adapthist_3d(temp_img / np.max(temp_img))
	
	# filter
	temp_lab[temp_lab > 0] = 1

	# local image label for scan
	test_Image.append(temp_img)
	test_Label.append(temp_lab)
	Barycenter.append(temp_mid)

### Utah Initialization -------------------------------------------------------------------------------------------------------------------------------------------
N_train_patients = 30	# number of patients to use from train set

os.chdir("/hpc/zxio506/Atria_Data/Utah_Bi_Atria")

# list all the files in training and testing sets
files = os.listdir()

### Utah Data ----------------------------------------------------------------------------------------------------------------------------------------------
# Train Data: loop through all training patients
for i in range(N_train_patients): # [5,6,7]: #

	print(str(i+1)+" Processing Train Set: "+files[i])
	
	pat_files = os.listdir(files[i])
	
	for j in range(len(pat_files)):
		
		# read in the MRI scan and contrast normalization
		patient_3DMRI_scan = load_nrrd(os.path.join(files[i],pat_files[j],'lgemri.nrrd'))
		patient_3DMRI_scan = equalize_adapthist_3d(patient_3DMRI_scan / np.max(patient_3DMRI_scan))
		
		# load LA endo (with PVs) 
		laendo = load_nrrd(os.path.join(files[i],pat_files[j],'laendo.nrrd'))//255
		
		# cavity labels
		lab_folder = os.path.join(files[i],pat_files[j],"CARMA_"+files[i][5:]+"_"+pat_files[j]+"_full")
		lab_files  = os.listdir(lab_folder)
		
		for n in range(len(lab_files)):
			
			# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
			temp_lab = cv2.imread(os.path.join(lab_folder,lab_files[n]),cv2.IMREAD_GRAYSCALE)
			temp_img = patient_3DMRI_scan[:,:,n]

			# add laendo
			temp_lab[temp_lab > 0] = 1

			# find the center of mass of the mask
			midpoint = ndimage.measurements.center_of_mass(temp_lab > 0)
		
			# crop to utah size
			x,y = temp_img.shape
			temp_img_temp = temp_img[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
			temp_lab_temp = temp_lab[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]

			# resize
			temp_img_temp = cv2.resize(temp_img_temp, (272, 272))
			temp_lab_temp = cv2.resize(temp_lab_temp, (272, 272))
			
			# local image label for scan
			train_Image.append(temp_img_temp)
			train_Label.append(temp_lab_temp)

# Test Data: loop through all training patients
for i in range(N_train_patients,len(files)):

	print(str(i+1)+" Processing Test Set: "+files[i])
	
	pat_files = os.listdir(files[i])
	
	for j in range(len(pat_files)):
		
		# read in the MRI scan and contrast normalization
		patient_3DMRI_scan = load_nrrd(os.path.join(files[i],pat_files[j],'lgemri.nrrd'))
		patient_3DMRI_scan = equalize_adapthist_3d(patient_3DMRI_scan / np.max(patient_3DMRI_scan))
		
		# crop shape so all outputs are 576 x 576
		x,y,z = patient_3DMRI_scan.shape
		patient_3DMRI_scan = patient_3DMRI_scan[(x//2-288):(x//2+288),(y//2-288):(y//2+288),:]
		
		# load LA endo (with PVs) 
		laendo = load_nrrd(os.path.join(files[i],pat_files[j],'laendo.nrrd'))//255
		
		# cavity labels
		lab_folder = os.path.join(files[i],pat_files[j],"CARMA_"+files[i][5:]+"_"+pat_files[j]+"_full")
		lab_files  = os.listdir(lab_folder)

		# loop through all the slices
		temp_img,temp_lab,temp_mid = np.zeros([272,272,44]),np.zeros([272,272,44]),np.zeros([44,2])
		for n in range(len(lab_files)):
			
			# load label (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
			temp_lab_temp = cv2.imread(os.path.join(lab_folder,lab_files[n]),cv2.IMREAD_GRAYSCALE)
			
			# add laendo
			temp_lab_temp[temp_lab_temp > 0] = 1
			
			# crop so all outputs are 576 x 576
			temp_img_temp = patient_3DMRI_scan[(x//2-288):(x//2+288),(y//2-288):(y//2+288),n]
			temp_lab_temp = temp_lab_temp[(x//2-288):(x//2+288),(y//2-288):(y//2+288)]
			
			# resize
			temp_img_temp = cv2.resize(temp_img_temp, (272, 272))
			temp_lab_temp = cv2.resize(temp_lab_temp, (272, 272))
			
			# find the center of mass of the mask
			temp_img[:,:,n] = temp_img_temp
			temp_lab[:,:,n] = temp_lab_temp
			temp_mid[n,:]   = ndimage.measurements.center_of_mass(temp_lab_temp > 0)

		# local image label for scan
		test_Image.append(temp_img)
		test_Label.append(temp_lab)
		Barycenter.append(temp_mid)
	
### Save Data ----------------------------------------------------------------------------------------------------------------------------------------------
os.chdir("/hpc/zxio506/2022_runs")

create_folder("UtahWaikato Test Set ROI")
create_folder("UtahWaikato Test Set ROI/log")

# Train Data
train_Image,train_Label = np.array(train_Image),np.array(train_Label)
train_Label[train_Label > 0] = 1

# encoding label to neural network output format
temp = np.empty(shape=[train_Label.shape[0],272,272,2])
for i in range(2):
	x = train_Label == i
	temp[:,:,:,i] = x

train_Image = np.reshape(train_Image,newshape=[-1,272,272,1])
train_Label = np.reshape(temp,newshape=[-1,272,272,2])

# Test Data
test_Image,test_Label,Barycenter = np.array(test_Image),np.array(test_Label),np.array(Barycenter)
test_Label[test_Label > 0] = 1
# create a HDF5 dataset
h5f = h5py.File('UtahWaikato Test Set ROI/Training.h5','w')
h5f.create_dataset("image",data=train_Image)
h5f.create_dataset("label",data=train_Label)
h5f.close()

# create a HDF5 dataset
h5f = h5py.File('UtahWaikato Test Set ROI/Testing.h5','w')
h5f.create_dataset("image",     data=test_Image)
h5f.create_dataset("label",     data=test_Label)
h5f.create_dataset("centroid",  data=Barycenter)
h5f.close()
