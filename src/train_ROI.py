import os
import h5py
import numpy as np
from sklearn.metrics import f1_score
from Utils import *

n1 = 272 # x
n2 = 272 # y

### Evaluation Function ------------------------------------------------------------------------------------------------------------------
def evaluate(data,CNN_model,log_path,mu=0,sd=1):

	# initialize output log file
	f1_scores = []
	
	# loop through all test patients
	for i in range(data["image"].shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([n1,n2,data["image"].shape[3]])
		
		for j in range(data["image"].shape[3]):

			# make prediction for slices with midpoints
			data_i = [(data["image"][i,:,:,j][:,:,None]-mu)/sd]
			data_o = CNN_model.predict(data_i)

			pred[:,:,j] = np.argmax(data_o,3)[0]
	
		# Evaluation (0 = background, 1 = RA+LA wall, 2 = RA endo, 3 = LA endo)
		true_flat,pred_flat = data["label"][i].flatten(),pred.flatten()
		
		temp = f1_score(true_flat,pred_flat,average=None)

		# store scores
		f1_scores.append(np.mean(temp))

	# overall score
	f = open(log_path,"a")
	f.write("\nOVERALL DSC AVEARGE = "+str(np.round(np.mean(np.array(f1_scores)),5))+"\n")
	f.write("\n\n")
	f.close()

	return(np.array(f1_scores))

### Computation Graph ------------------------------------------------------------------------------------------------------------------
model = AtriaNet_ROI(n1, n2)

### Training ------------------------------------------------------------------------------------------------------------------
# set main directory
os.chdir("UtahWaikato Test Set ROI")

# set up log file
log_path = "log/log.txt"

# load training data and apply mean and standard deviation normalization
train_data,test_data = h5py.File("Training.h5","r"),h5py.File("Testing.h5","r")
train_mean,train_sd  = np.mean(train_data["image"]),np.std(train_data["image"])

# keep track of best dice score
best_DSC = 0
f = open(log_path,"w");f.write("mean: "+str(np.round(train_mean,5))+"\t std: "+str(np.round(train_sd,5)));f.close()

for n in range(25):
	
	f = open(log_path,"a");f.write("\n"+"-"*50+" Epoch "+str(n+1));f.close()

	# online augmentation
	train_image,train_label,train_mean,train_sd = online_augmentation(train_data["image"],train_data["label"])
	
	# run 1 epoch
	model.fit(train_image,train_label,n_epoch=1,show_metric=True,batch_size=8,shuffle=True)
	
	# evaluate current performance
	DSCs = evaluate(test_data,model,log_path,train_mean,train_sd)
	
	# if the model is currently the best
	if np.mean(DSCs) > best_DSC:
		best_DSC = np.mean(DSCs)
		model.save("log/LARAmodel_UtahWaikato")
