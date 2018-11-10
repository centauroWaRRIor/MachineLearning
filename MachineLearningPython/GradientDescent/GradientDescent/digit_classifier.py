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
		#self.classifier = Batch_Gradient_Descent(9, True)

	def run_until_convergence(self, number_epoch, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		# randomize the training set
		random.shuffle(inputs_vector) 

		for i in range(number_epoch):
			print "Training Epoch # %d" % i
			self.train_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
			self.evaluate_accuracy_one_epoch(inputs_vector, ground_truth_labels)


	def train_one_epoch(self, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		#for perceptron in self.perceptrons:
		#	perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)
		self.classifier.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		error = self.classifier.calc_error(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		print "error = %f" % error

	def evaluate_accuracy_one_epoch(self, inputs_vector, ground_truth_labels):

		self.scorer.reset()
		for inputs, label in zip(inputs_vector, ground_truth_labels):
			prediction = self.classifier.predict(inputs, True)
			binary_label = 0
			if self.classifier.predict_label == label:
				binary_label = 1
			self.scorer.record_result(binary_label, prediction)
			#print "predicted %d vs truth %d" % (prediction, binary_label) 

		print "accuracy = %f" % self.scorer.get_accuracy()

