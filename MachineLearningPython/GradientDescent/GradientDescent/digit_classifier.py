from score import Classifier_Score
from gradient_descent import Batch_Gradient_Descent

class Digit_Classifier:
	"""10 GD classifiers, each one trained to recognize one digit"""
		
	def __init__(self, perceptron_type="Placeholder"):
		self.scorer = Classifier_Score()
		self.classifiers = []
		for i in range(10):
			self.classifiers.append(Batch_Gradient_Descent(i, True))

	def is_converged(self):
		# my convergence rule is the following: all the classifiers need to converge
		# before we can consider the overall system as converged
		convergence_consensus = True
		for classifier in self.classifiers:
			convergence_consensus = convergence_consensus and classifier.is_converged
		return convergence_consensus


	def run_until_convergence(self, l_rate, lambda_value,
						      inputs_vector_train, ground_truth_labels_train,
							  inputs_vector_test, ground_truth_labels_test):

		# the training set is assumed to be randomized at this point
		epoch_number = 0
		while not self.is_converged():
			classifier_index = 0
			average_loss = 0.0
			# train and test all digit classifiers
			for classifier in self.classifiers:
				# training is skipped if given classifier has converged
				loss = self.train_one_epoch(classifier_index, inputs_vector_train, ground_truth_labels_train, l_rate, lambda_value)
				average_loss += loss
				training_accuracy = self.evaluate_classifier_accuracy_one_epoch(classifier_index, inputs_vector_train, ground_truth_labels_train)
				#test_accuracy = self.evaluate_classifier_accuracy_one_epoch(classifier_index, inputs_vector_test, ground_truth_labels_test)
				print "classifier index: %d, epoch: %d, Training Loss: %0.2f, Training Accuracy: %0.2f" % \
					(classifier_index, epoch_number, loss, training_accuracy)
				classifier_index += 1

			average_loss /= len(self.classifiers)
			training_accuracy = self.evaluate_system_accuracy_one_epoch(inputs_vector_train, ground_truth_labels_train)
			test_accuracy = self.evaluate_system_accuracy_one_epoch(inputs_vector_test, ground_truth_labels_test)
			print "============================================================================="
			print "epoch: %d Training Loss: %0.2f, Training Accuracy: %0.2f, Test Accuracy: %0.2f" % \
				(epoch_number, average_loss, training_accuracy, test_accuracy)
			print "============================================================================="
			epoch_number += 1


	def train_one_epoch(self, classifier_index, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		#for perceptron in self.perceptrons:
		#	perceptron.train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate)
		self.classifiers[classifier_index].train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		error = self.classifiers[classifier_index].calc_error_one_epoch(inputs_vector, ground_truth_labels, l_rate, lambda_value)
		return error

	def evaluate_classifier_accuracy_one_epoch(self, classifier_index, inputs_vector, ground_truth_labels):

		self.scorer.reset()
		for inputs, label in zip(inputs_vector, ground_truth_labels):
			prediction = self.classifiers[classifier_index].predict(inputs, True)
			binary_label = 0
			if self.classifiers[classifier_index].predict_label == label:
				binary_label = 1
			self.scorer.record_result(binary_label, prediction)
			#print "predicted %d vs truth %d" % (prediction, binary_label) 
		return self.scorer.get_accuracy()

	def evaluate_system_accuracy_one_epoch(self, inputs_vector, ground_truth_labels):

		self.scorer.reset()
		
		for inputs, label in zip(inputs_vector, ground_truth_labels):
			
			classifiers_predictions = []
			for classifier in self.classifiers:
				# collect each digit classifier prediction in raw format (without the rounding)
				classifiers_predictions.append(classifier.predict(inputs, False))

			lowest_prediction = 1.0
			prediction = 0
			for i in range(len(classifiers_predictions)):
				if classifiers_predictions[i] < lowest_prediction:
					lowest_prediction = classifiers_predictions[i]
					prediction = i
			
			self.scorer.record_result(label, prediction)
			#print "predicted %d vs truth %d" % (prediction, binary_label) 
		return self.scorer.get_accuracy()