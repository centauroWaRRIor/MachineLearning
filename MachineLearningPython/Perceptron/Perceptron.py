
import sys
import struct
import random
import numpy as np


class MNIST_Datastream:
	"""Simulates an infinite stream of examples"""

	def __init__(self, images_filename, labels_filename):

		# open file
		self.images_file = open(images_filename, 'rb')
		self.labels_file = open(labels_filename, 'rb')

		# extract magic number
		self.images_file.seek(0)
		self.magic_number = struct.unpack('>4B', self.images_file.read(4))
		self.labels_file.seek(0)
		self.magic_number = struct.unpack('>4B', self.labels_file.read(4))

		#32-bit integer (4 bytes)
		self.images_file.seek(4)
		self.num_images = struct.unpack('>I', self.images_file.read(4))[0]
		print "num images: ", self.num_images
		self.num_rows = struct.unpack('>I', self.images_file.read(4))[0]
		print "num rows: ", self.num_rows
		self.num_columns = struct.unpack('>I', self.images_file.read(4))[0]
		print "num columns: ", self.num_columns

		self.labels_file.seek(4)
		assert(self.num_images == struct.unpack('>I', self.labels_file.read(4))[0])

		self.label_bytes = 1 
		self.image_bytes = self.num_rows * self.num_columns # one byte per grid entry


	def __del__(self):

		self.images_file.close()
		self.labels_file.close()

	def get_label(self, image_index):

		file_offset = 8 + image_index * self.label_bytes
		self.labels_file.seek(file_offset) # absolute offset
		label = struct.unpack('>'+'B' * self.label_bytes, self.labels_file.read(self.label_bytes))[0]
		# returns python array
		return label

	def get_image(self, image_index):

		file_offset = 16 + image_index * self.image_bytes
		self.images_file.seek(file_offset) # absolute offset
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * self.image_bytes, self.images_file.read(self.image_bytes))).reshape(self.num_rows, self.num_columns)
		# returns python array
		return image_numpy_array.tolist()


	def get_rounded_image(self, image_index):

		image = self.get_image(image_index)
		for i in range(self.num_rows):
			for j in range(self.num_columns):
				image[i][j] = int(round(image[i][j]/255.0))
		return image

	def get_1d_image(self, image_index):

		file_offset = 16 + image_index * self.image_bytes
		self.images_file.seek(file_offset) # absolute offset
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * self.image_bytes, self.images_file.read(self.image_bytes)))
		# returns python array
		return image_numpy_array.tolist()


	def get_rounded_1d_image(self, image_index):

		image = self.get_1d_image(image_index)
		for i in range(self.num_rows * self.num_columns):
			image[i] = int(round(image[i]/255.0))
		return image


	def ascii_show(self, image):
		for y in image:
			row = ""
			for x in y:
				row += '{0: <4}'.format(x)
			print row

class Vanilla_Perceptron:
	""" Vanilla implementation of a perceptron with 784 features + bias
	The 784 inputs come directly from MNIST hand written images once the vector
	has been flattened"""
	
	def __init__(self, predict_label,init_random_weights=False):
		self.scorer = Classifier_Score()
		self.num_features = 284
		self.weights = []
		self.predict_label = predict_label # this is the handwritten digit this perceptron is going to learn to predict
		# initialize the weights
		for i in range(self.num_features + 1): # + 1 for the bias term which is treated as just another feature
			if init_random_weights:
				weights.append(random.uniform(-1.0, 1.0))
			else:
				weights.append(0.0)

	def predict(inputs):
		w_dot_x = 0.0
		for i in range(self.num_features + 1): # + 1 for the bias term which is treated as just another feature
			w_dot_x += inputs[i] * self.weights[i] 

		if w_dot_x >= 0.0:
			return 1.0
		else:
			return 0.0

	def train_weights_one_epoch(inputs_vector, ground_truth_labels, l_rate=1.00):

		assert(len(inputs_vector) == len(ground_truth_labels))

		for inputs, label in zip(inputs_vector, ground_truth_labels):
			prediction = predict(inputs) # get predicted classificaion, 0 or 1
			# label is a digit from 0 to 9 so need to normalize
			normalized_label = 0
			if self.predict_label == normalized_label:
				normalized_label = 1
			y_hat = normalized_label - prediction
			scorer.record_result(self, normalized_label, prediction)
			error = abs(y_hat)	# get error from real classification 0 or 1 
			
			for i in range(self.num_features + 1): 	# calculate new weight for each node
				self.weights[i] = self.weights[i] + (error * l_rate * y_hat * inputs[i]) 

class Classifier_Score:
	"""Book keeping object for tracking F1 and Accuracy scores
	"""

	class F1_Score:

		def __init__(self, label, ground_truth_array, predictions_array):
			"""The learner can make two kind of mistakes
			"""
			assert(len(ground_truth_array) == len(predictions_array))
			self.false_positive = 0
			self.false_negative = 0
			self.true_positive = 0
			#self.true_negative = 0
			self.label = label

			self.ground_truth_array = ground_truth_array
			self.predictions_array = predictions_array

			for i in range(len(ground_truth_array)):
				prediction = predictions_array[i]
				ground_truth = ground_truth_array[i]
				if prediction == label:
					if ground_truth == prediction:
						self.true_positive += 1
					else:
						self.false_positive += 1
				elif ground_truth == label:
					self.false_negative += 1
				else:
					#self.true_negative += 1
					continue


		def get_precision(self):
			"""When we predicted the rare class how often are we right
			"""
			if self.true_positive == 0:
				return 0.0
			else:
				return self.true_positive / float(self.true_positive + self.false_positive)

		def get_recall(self):
			"""Out of all the instances  of the rare class, how many did we 
			catch?
			"""
			if self.true_positive == 0:
				return 0.0
			else: 
				return self.true_positive / float(self.true_positive + self.false_negative)

		def get_f1_score(self):
			P = self.get_precision()
			R = self.get_recall()
			if P + R == 0.0:
				return 0.0
			else:
				return ( 2 * P * R ) / (P + R)


	def __init__(self):
		self.reset()

	def reset(self):
		self.correct_classifications = 0
		self.total_classifications = 0
		self.ground_truth_array = []
		self.predictions_array = []
		self.label_count_dict = {}

	def record_result(self, ground_truth, prediction):
		
		if ground_truth == prediction:
			self.correct_classifications += 1
		self.total_classifications += 1

		# book keeping for calculating the F1 score of each label later
		self.ground_truth_array.append(ground_truth)
		self.predictions_array.append(prediction)

		if ground_truth not in self.label_count_dict:
			self.label_count_dict[ground_truth] = 0
		else:
			self.label_count_dict[ground_truth] += 1

		if prediction not in self.label_count_dict:
			self.label_count_dict[prediction] = 0
		else:
			self.label_count_dict[prediction] += 1

	def get_accuracy(self):
		return self.correct_classifications / float(self.total_classifications)

	def get_macro_F1_score(self):
		"""TA recommended using the average of all the labels instead of 
		using the weighted average above"""

		total_labels_present = len(self.label_count_dict.keys())

		labels_f1_sum = 0
		for label in self.label_count_dict.keys():
			label_f1_score = self.F1_Score(label, self.ground_truth_array, self.predictions_array)
			labels_f1_sum += label_f1_score.get_f1_score()

		return labels_f1_sum / float(total_labels_present)

def main():

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
	data_stream = MNIST_Datastream(images_filename, labels_filename)
	#image_python_array = data_stream.get_image(33)
	#data_stream.ascii_show(image_python_array)
	#print data_stream.get_label(33)
	#print "--------------"
	#image_python_array = data_stream.get_rounded_image(33)
	#data_stream.ascii_show(image_python_array)
	
	#image_python_array = data_stream.get_rounded_1d_image(33)
	#print image_python_array
	
	f1_scoring = Classifier_Score()
	y_true = [0, 1, 2, 0, 1, 2]
	y_pred = [0, 2, 1, 0, 0, 1]
	for true, pred in zip(y_true, y_pred):
		f1_scoring.record_result(true, pred)
	print "F1-Score: ", f1_scoring.get_macro_F1_score();

	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio