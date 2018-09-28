from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np

def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0

# each matrix row: up to last row = inputs, last row = y (classification)
def accuracy(matrix,weights):
	num_correct = 0.0
	preds       = []
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		preds.append(pred)
		if pred==matrix[i][-1]: num_correct+=1.0 
	print("Predictions:",preds)
	return num_correct/float(len(matrix))

# each matrix row: up to last row = inputs, last row = y (classification)
def train_weights(matrix,weights,nb_epoch=10,l_rate=1.00,do_plot=False,stop_early=True,verbose=True):
	for epoch in range(nb_epoch):
		cur_acc = accuracy(matrix,weights)
		print("\nEpoch %d \nWeights: "%epoch,weights)
		print("Accuracy: ",cur_acc)
		
		if cur_acc==1.0 and stop_early: break 
		#if do_plot and len(matrix[0])==4: plot(matrix,weights) # if 2D inputs, excluding bias
		if do_plot: plot(matrix,weights,title="Epoch %d"%epoch)
		
		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],weights) # get predicted classificaion
			error      = matrix[i][-1]-prediction		 # get error from real classification
			if verbose: sys.stdout.write("Training on data at index %d...\n"%(i))
			for j in range(len(weights)): 				 # calculate new weight for each node
				if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j,weights[j]))
				weights[j] = weights[j]+(l_rate*error*matrix[i][j]) 
				if verbose: sys.stdout.write("%0.5f\n"%(weights[j]))

	#if len(matrix[0])==4: plot(matrix,weights) # if 2D inputs, excluding bias
	plot(matrix,weights,title="Final Epoch")
	return weights 

def main():

	nb_epoch		= 10
	l_rate  		= 1.0
	plot_each_epoch	= False
	stop_early 		= True

	part_A = True  

	if part_A: # 3 inputs (including single bias input), 3 weights

				# 	Bias 	i1 		i2 		y
		matrix = [	[1.00,	0.08,	0.72,	1.0],
					[1.00,	0.10,	1.00,	0.0],
					[1.00,	0.26,	0.58,	1.0],
					[1.00,	0.35,	0.95,	0.0],
					[1.00,	0.45,	0.15,	1.0],
					[1.00,	0.60,	0.30,	1.0],
					[1.00,	0.70,	0.65,	0.0],
					[1.00,	0.92,	0.45,	0.0]]
		weights= [	 0.20,	1.00,  -1.00		] # initial weights specified in problem

	else: # 2 inputs (including single bias input), 2 weights
		
		nb_epoch = 1000

				# 	Bias 	i1 		y
		matrix = [	[1.00,	0.08,	1.0],
					[1.00,	0.10,	0.0],
					[1.00,	0.26,	1.0],
					[1.00,	0.35,	0.0],
					[1.00,	0.45,	1.0],
					[1.00,	0.60,	1.0],
					[1.00,	0.70,	0.0],
					[1.00,	0.92,	0.0]]
		weights= [	 0.20,	1.00		] # initial weights specified in problem

	train_weights(matrix,weights=weights,nb_epoch=nb_epoch,l_rate=l_rate,do_plot=plot_each_epoch,stop_early=stop_early)


if __name__ == '__main__':
	main()
