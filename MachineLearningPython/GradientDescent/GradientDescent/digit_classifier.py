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

	def run_until_convergence(self, number_epoch, l_rate, lambda_value,
						      inputs_vector_train, ground_truth_labels_train,
							  inputs_vector_test, ground_truth_labels_test):

		# randomize the training set
		#random.shuffle(inputs_vector)  no so fast cowboy! this is possibly a bug, If I reshuffle the inputs vector then I lose the parity with the ground truths vector

		for i in range(number_epoch):
			loss = self.train_one_epoch(inputs_vector_train, ground_truth_labels_train, l_rate, lambda_value)
			training_accuracy = self.evaluate_accuracy_one_epoch(inputs_vector_train, ground_truth_labels_train)
			test_accuracy = self.evaluate_accuracy_one_epoch(inputs_vector_test, ground_truth_labels_test)
			print "epoch %d: Training Loss: %0.2f, Testing Accuracy: %0.2f, Test Accuracy: %0.2f" % \
				(i, loss, training_accuracy, test_accuracy)

	def train_one_epoch(self, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		#for perceptron in self.perceptrons:
		#	perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)
		self.classifier.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		error = self.classifier.calc_error(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		return error

	def evaluate_accuracy_one_epoch(self, inputs_vector, ground_truth_labels):

		self.scorer.reset()
		for inputs, label in zip(inputs_vector, ground_truth_labels):
			prediction = self.classifier.predict(inputs, True)
			binary_label = 0
			if self.classifier.predict_label == label:
				binary_label = 1
			self.scorer.record_result(binary_label, prediction)
			#print "predicted %d vs truth %d" % (prediction, binary_label) 
		return self.scorer.get_accuracy()

