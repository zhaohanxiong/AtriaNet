import os
import h5py
import scipy.io
import numpy as np
from Utils import *

n1 = 272 # x
n2 = 272 # y

### Evaluation Function ------------------------------------------------------------------------------------------------------------------
def evaluate(data,CNN_model,log_path,mu=0,sd=1):

	# set up 
	test_image,test_label = data["label"],data["awt"]
	
	# initialize output log file
	mse_masked_score,mse_scores,dice_score = [],[],[]
	
	# loop through all test patients
	for i in range(test_image.shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,44])
		
		for j in range(test_image.shape[3]):
			
			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]
			
			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):
				
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(test_image[i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = data_o[0,:,:,0]

		# mask out the non atrial wall pixels in the prediction
		pred[test_label[i] == 0] = 0
		
		# Evaluation
		mse        = np.mean((test_label[i] - pred)**2)
		mse_masked = np.mean((test_label[i][test_label[i] > 0] - pred[test_label[i] > 0])**2)
		
		t,p  = test_label[i] > 0,pred > 0
		dice = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))
		
		# store scores
		mse_masked_score.append(mse_masked)
		mse_scores.append(mse)
		dice_score.append(dice)

	# overall score
	f = open(log_path,"a")
	f.write("\nOVERALL MSE MASKED AVEARGE = "+str(np.round(np.mean(np.array(mse_masked_score)),3))+"\n")
	f.write("\nOVERALL MSE AVEARGE        = "+str(np.round(np.mean(np.array(mse_scores)),3))+"\n")
	f.write("\nOVERALL DSC AVEARGE        = "+str(np.round(np.mean(np.array(dice_score)),3))+"\n")
	f.write("\n\n")
	f.close()

	return(np.array(mse_masked_score))

def save_best(data,CNN_model,mu=0,sd=1):

	print("\nSaving Best Outputs...\n")

	# set up 
	test_image,test_label = data["label"],data["awt"]

	# loop through all test patients
	for i in range(test_image.shape[0]):

		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,44])

		for j in range(test_image.shape[3]):

			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]

			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):

				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(test_image[i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = data_o[0,:,:,0]

		# mask out the non atrial wall pixels in the prediction
		pred[test_label[i] == 0] = 0

		# save to output
		scipy.io.savemat("Prediction Sample/test"+"{0:03}".format(i)+".mat",mdict={"input_seg": test_image[i],
		                                                                           "true":      test_label[i],
																				   "pred":      pred,
																				   "lgemri":    data["image"][i]})

### Computation Graph ------------------------------------------------------------------------------------------------------------------
model = AtriaNet_AWT(n1, n2)

### Training ------------------------------------------------------------------------------------------------------------------
# set main directory
os.chdir("UtahAWT Test Set")

# set up log file
log_path = "log/log.txt"

# load training data (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo)
train_data,test_data = h5py.File("Training.h5","r"),h5py.File("Testing.h5","r")

# preprocess input and output data
train_image,train_label = train_data["label"],train_data["awt"]
train_image = np.argmax(train_image,3)[:,:,:,None] # LA+RA endo+wall as input

# keep track of best dice score
f = open(log_path,"w");f.close()
best_mse = 1000

for n in range(100):
	
	f = open(log_path,"a");f.write("-"*50+" Epoch "+str(n+1)+"\n");f.close()

	# run 1 epoch
	model.fit(train_image,train_label,n_epoch=1,show_metric=True,batch_size=16,shuffle=True)
	
	# evaluate current performance
	MSEs = evaluate(test_data,model,log_path)
	
	# if the model is currently the best
	if np.mean(MSEs) < best_mse:
		best_mse = np.mean(MSEs)
		save_best(test_data,model)
		model.save("log/AWTmodel_Utah")
