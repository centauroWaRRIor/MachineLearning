
import sys
import struct
import random
import numpy as np

class Vanilla_Perceptron:
	""" Vanilla implementation of a perceptron with 784 features + bias
	The 784 inputs come directly from MNIST hand written images once the vector
	has been flattened"""
	
	def __init__(self, predict_label, init_random_weights=False):
		self.num_features = 785
		self.weights = []
		self.predict_label = predict_label # this is the handwritten digit this perceptron is going to learn to predict
		# initialize the weights
		for i in range(self.num_features):
			if init_random_weights:
				self.weights.append(random.uniform(-1.0, 1.0))
			else:
				self.weights.append(0.0)

	def predict(self, inputs):

		assert(len(inputs) == len(self.weights))

		w_dot_x = 0.0
		for i in range(self.num_features):
			w_dot_x += inputs[i] * self.weights[i] 

		if w_dot_x >= 0.0:
			return 1
		else:
			return 0

	def predict_w_dot_x(self, inputs):

		assert(len(inputs) == len(self.weights))
		
		w_dot_x = 0.0
		for i in range(self.num_features):
			w_dot_x += inputs[i] * self.weights[i] 

		if w_dot_x >= 0.0:
			return w_dot_x
		else:
			return 0.0

	def train_weights_one_epoch(self, inputs_vector, ground_truth_labels, l_rate=1.00):

		assert(len(inputs_vector) == len(ground_truth_labels))

		for inputs, label in zip(inputs_vector, ground_truth_labels):

			prediction = self.predict(inputs) # get predicted classificaion, 0 or 1
			# label is a digit from 0 to 9 so need to normalize
			binary_label = 0
			if self.predict_label == label:
				binary_label = 1
			y_hat = binary_label - prediction		
			error = abs(y_hat)	# get error from real classification 0 or 1 
			
			for i in range(self.num_features):
				self.weights[i] = self.weights[i] + (error * l_rate * y_hat * inputs[i]) 

class Digit_Classifier_Vanilla:
	"""10 vanilla perceptrons, each one trained to recognize one digit"""
		
	def __init__(self):
		self.scorer = Classifier_Score()
		self.perceptrons = []
		for i in range(10):
			self.perceptrons.append(Vanilla_Perceptron(i))


	def train(self, number_epoch, inputs_vector, ground_truth_labels, l_rate=1.00):
		for i in range(number_epoch):
			print ("Epoch# ", i)
			for perceptron in self.perceptrons:
				perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)

	def predict(self, inputs_vector, ground_truth_labels):

		self.scorer.reset()
		for inputs, label in zip(inputs_vector, ground_truth_labels):
			largest_w_dot_x = 0
			largest_w_dot_x_index = 0 # by default predict 0
			for i in range(10):
				w_dot_x = self.perceptrons[i].predict_w_dot_x(inputs)
				print ("predicted w_dot_x: ", w_dot_x)
				if w_dot_x > largest_w_dot_x:
					largest_w_dot_x = w_dot_x
					largest_w_dot_x_index = i
			# prediction is stored in largest_w_dot_x_index
			print ("predicted: ", largest_w_dot_x_index, "expected: ", label)
			self.scorer.record_result(label, largest_w_dot_x_index)

		# report F1-Score
		print "F1-score: ", self.scorer.get_macro_F1_score()


def main():

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
	training_data_stream = MNIST_Datastream(images_filename, labels_filename)

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
	test_data_stream = MNIST_Datastream(images_filename, labels_filename)

	#image_python_array = training_data_stream.get_image(33)
	#training_data_stream.ascii_show(image_python_array)
	#print training_data_stream.get_label(33)
	#print "--------------"
	#image_python_array = training_data_stream.get_rounded_image(33)
	#training_data_stream.ascii_show(image_python_array)
	
	#image_python_array = training_data_stream.get_rounded_1d_image(33)
	#print image_python_array
	
	#f1_scoring = Classifier_Score()
	#y_true = [0, 1, 2, 0, 1, 2]
	#y_pred = [0, 2, 1, 0, 0, 1]
	#for true, pred in zip(y_true, y_pred):
	#	f1_scoring.record_result(true, pred)
	#print "F1-Score: ", f1_scoring.get_macro_F1_score()

	inputs_vector = []
	ground_truth_labels = []

	for i in range(500):
		feature_inputs = training_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(training_data_stream.get_label(i))

	vanilla_classifier = Digit_Classifier_Vanilla()
	vanilla_classifier.train(50, inputs_vector, ground_truth_labels, 0.001)

	# Predict the training data
	print "Predict the training data: "
	vanilla_classifier.predict(inputs_vector, ground_truth_labels)

	print "Predict the test data: "
	inputs_vector = []
	ground_truth_labels = []
	for i in range(500):
		feature_inputs = test_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(test_data_stream.get_label(i))
	vanilla_classifier.predict(inputs_vector, ground_truth_labels)

	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio