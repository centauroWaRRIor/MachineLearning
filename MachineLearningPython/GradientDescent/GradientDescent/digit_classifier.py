from score import Classifier_Score
from gradient_descent import Batch_Gradient_Descent
import random

class Digit_Classifier:
	"""10 GD classifiers, each one trained to recognize one digit"""
		
	def __init__(self, perceptron_type="Placeholder"):
		self.scorer = Classifier_Score()
		#self.perceptrons = []
		#for i in range(10):
		#	if perceptron_type == "Vanilla_Perceptron":
		#		self.perceptrons.append(Vanilla_Perceptron(i))
		#	elif perceptron_type == "Average_Perceptron":
		#		self.perceptrons.append(Average_Perceptron(i))
		#	elif perceptron_type == "Winnow_Perceptron":
		#		self.perceptrons.append(Vanilla_Winnow(i))
		self.classifier = Batch_Gradient_Descent(5, True)


	def train(self, number_epoch, inputs_vector, ground_truth_labels, l_rate, lambda_value, verbose=False):

		# randomize the training set
		random.shuffle(inputs_vector) 

		for i in range(number_epoch):
			if verbose:
				print "Training Epoch # %d" % i
			#for perceptron in self.perceptrons:
			#	perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)
			self.classifier.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
			error = self.classifier.calc_error(inputs_vector, ground_truth_labels, l_rate, lambda_value)
			print error

	#def evaluate_f1_performance(self, inputs_vector, ground_truth_labels):

	#	self.scorer.reset()
	#	for inputs, label in zip(inputs_vector, ground_truth_labels):
	#		largest_w_dot_x = 0
	#		largest_w_dot_x_index = 0 # by default predict 0
	#		for i in range(10):
	#			w_dot_x = self.perceptrons[i].predict_w_dot_x(inputs)
	#			#print ("predicted w_dot_x: ", w_dot_x)
	#			if w_dot_x > largest_w_dot_x:
	#				largest_w_dot_x = w_dot_x
	#				largest_w_dot_x_index = i
	#		# prediction is stored in largest_w_dot_x_index
	#		#print ("predicted: ", largest_w_dot_x_index, "expected: ", label)
	#		self.scorer.record_result(label, largest_w_dot_x_index)

	#	# return F1-Score
	#	return self.scorer.get_macro_F1_score()

