from score import Classifier_Score
from gradient_descent import Batch_Gradient_Descent

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

		#self.classifier = Batch_Gradient_Descent(1, True)
		#self.classifier = Batch_Gradient_Descent(2, True)
		#self.classifier = Batch_Gradient_Descent(3, True)
		#self.classifier = Batch_Gradient_Descent(4, True)
		self.classifier = Batch_Gradient_Descent(5, True)
		#self.classifier = Batch_Gradient_Descent(6, True)
		#self.classifier = Batch_Gradient_Descent(7, True)
		#self.classifier = Batch_Gradient_Descent(8, True)
		#self.classifier = Batch_Gradient_Descent(9, True)

	def is_converged(self):

		return self.classifier.is_converged

	def run_until_convergence(self, l_rate, lambda_value,
						      inputs_vector_train, ground_truth_labels_train,
							  inputs_vector_test, ground_truth_labels_test):

		# the training set is assumed to be randomized at this point
		i = 0
		while not self.is_converged():
			loss = self.train_one_epoch(inputs_vector_train, ground_truth_labels_train, l_rate, lambda_value)
			training_accuracy = self.evaluate_accuracy_one_epoch(inputs_vector_train, ground_truth_labels_train)
			test_accuracy = self.evaluate_accuracy_one_epoch(inputs_vector_test, ground_truth_labels_test)
			print "epoch %d: Training Loss: %0.2f, Testing Accuracy: %0.2f, Test Accuracy: %0.2f" % \
				(i, loss, training_accuracy, test_accuracy)
			i += 1


	def train_one_epoch(self, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		#for perceptron in self.perceptrons:
		#	perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)
		self.classifier.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		error = self.classifier.calc_error_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
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

