import random
import math

class Batch_Gradient_Descent:
	""" Vanilla implementation of a batch GD with 784 features + bias.
	The 784 inputs come directly from MNIST hand written images once the vector
	has been flattened"""
	
	def __init__(self, predict_label, init_random_weights=False):
		self.num_features = 785
		self.weights = []
		self.delta_weights = []
		self.predict_label = predict_label # this is the handwritten digit this classifier is going to learn to predict
		# initialize the weights
		for i in range(self.num_features):
			if init_random_weights:
				self.weights.append(random.uniform(-1.0, 1.0))
			else:
				self.weights.append(0.0)
			self.delta_weights.append(0.0)

	def sigmoid_activation(self, z):
		return 1.0 / (1.0 + math.exp(-z))

	def predict(self, inputs, predict_activation):
		""" If predict activation is false, this function
		will return the raw value after going through the 
		sigmoid"""
		assert(len(inputs) == len(self.weights))
		# compute the prediction by taking the dot product of the
		# current feature vector with the weight matrix W
		w_dot_x = 0.0
		for i in range(self.num_features):
			w_dot_x += inputs[i] * self.weights[i] 

		# the sigmoid function is defined over the range y=[0, 1],
		raw_activation = self.sigmoid_activation(w_dot_x)

		if not predict_activation:
			return raw_activation

		if raw_activation < 0.5:
			return 0
		else:
			return 1

	def calc_error(self, inputs_vector, ground_truth_labels, l_rate, lambda_value): 
		assert(len(inputs_vector) == len(ground_truth_labels))

		# precalculate regularization term
		regularization_term = 0
		for i in range(self.num_features):
			regularization_term += self.weights[i] * self.weights[i]
		regularization_term *= lambda_value / (2.0 * float(len(inputs_vector)))

		error = 0
		for inputs, label in zip(inputs_vector, ground_truth_labels):

			prediction = self.predict(inputs, False) # get predicted raw classificaion from sigmoid
			# label is a digit from 0 to 9 so need to normalize
			binary_label = 0
			if self.predict_label == label:
				binary_label = 1

			# compute loss function Err(w)=-\sum_i {y^{i}\log(g(w, x_i)) + (1-y^{i})\log(1-g(w, x_i))}
			error += -(binary_label * math.log(prediction) + (1-binary_label)* math.log(prediction))
		error += regularization_term
		error /= float(len(inputs_vector))
		return error

	def train_weights_one_epoch(self, inputs_vector, ground_truth_labels, l_rate, lambda_value):

		assert(len(inputs_vector) == len(ground_truth_labels))

		# zero out the delta weights
		for i in range(self.num_features):
			self.delta_weights[i] = 0.0

		for inputs, label in zip(inputs_vector, ground_truth_labels):

			prediction = self.predict(inputs, False) # get predicted raw classificaion from sigmoid
			# label is a digit from 0 to 9 so need to normalize
			binary_label = 0.0
			if self.predict_label == label:
				binary_label = 1.0
			
			# compute the gradients and regularization term
			for j in range(self.num_features):
				# gradient = x_i^j (g(z) - y^i)
				gradient_j = inputs[j] * (prediction - binary_label)
				self.delta_weights[j] = self.delta_weights[j] + l_rate * (gradient_j)

		# update weights
		for j in range(self.num_features):
			regularization_term = (lambda_value * self.weights[j] / float(len(inputs_vector)))
			self.weights[j] += self.delta_weights[j]/float(len(inputs_vector)) + regularization_term