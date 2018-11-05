class Vanilla_Perceptron:
	""" Vanilla implementation of a perceptron with 784 features + bias.
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
			y_hat = binary_label - prediction # decide whether or not we need promotion or demotion	
			error = abs(y_hat)	# get error from real classification 0 or 1 
			
			for i in range(self.num_features):
				self.weights[i] = self.weights[i] + (error * l_rate * y_hat * inputs[i])						