import csv
import numpy
from random import shuffle
from KNN import KNN_Classifier
from ID3 import DecisionTree_ID3

class DataSet(object):
	def __init__(self):
		# Organizing the data as a list of dictionaries
		# each dictionary contains feature, value for easy
		# retrieval
		self.list_dict = []
		self.classification_label = None

	def load(self, file_name, classification_label, should_shuffle=True):

		self.classification_label = classification_label
		header_parsed = False
		label_keys = []
		with open(file_name, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				values = line[0].split(';')
				# first line contains the keys for the dictionary
				if not header_parsed:
					for value in values:
						label_keys.append(value.replace('\"', "")) # clean up the label
					# assuming first row always contains the feature names
					header_parsed = True
				else:
					if len(values) != len(label_keys):
						raise Exception("Bad input file")
					dict = {}
					# store the data into data structure of list of dicts
					# store everything as floats and "quality" as int 
					for j in range(len(label_keys)):
						if label_keys[j] == classification_label:
							dict[label_keys[j]] = int(values[j])
						else:
							dict[label_keys[j]] = float(values[j])
					self.list_dict.append(dict)

		if should_shuffle:
			shuffle(self.list_dict)
		
		# quickly verify my data structure has been built correctly
		#for j in range(100):
		#	print self.list_dict[j]["fixed acidity"], \
		#		self.list_dict[j]["volatile acidity"], \
		#		self.list_dict[j]["citric acid"], \
		#		self.list_dict[j]["residual sugar"], \
		#		self.list_dict[j]["chlorides"], \
		#		self.list_dict[j]["free sulfur dioxide"], \
		#		self.list_dict[j]["total sulfur dioxide"], \
		#		self.list_dict[j]["density"], \
		#		self.list_dict[j]["pH"], \
		#		self.list_dict[j]["sulphates"], \
		#		self.list_dict[j]["alcohol"], \
		#		self.list_dict[j]["quality"] \


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

		
		#if prediction not in self.label_count_dict:
		#	self.label_count_dict[prediction] = 0
		#else:
		#	self.label_count_dict[prediction] += 1

	def get_accuracy(self):
		return self.correct_classifications / float(self.total_classifications)

	def get_weighted_F1_score(self):

		weighted_f1_score = 0.0
		total_label_count = 0

		for label in self.label_count_dict.keys():
			total_label_count += self.label_count_dict[label]

		for label in self.label_count_dict.keys():
			label_f1_score = self.F1_Score(label, self.ground_truth_array, self.predictions_array)
			weight = self.label_count_dict[label] / float(total_label_count)
			weighted_f1_score += weight * label_f1_score.get_f1_score()
		
		return weighted_f1_score


def k_FoldValidation(k, classifier, rows, classification_label):
     
	dataset_size = len(rows)
	if k > dataset_size:
		raise Exception("Bad k-fold argument given size of data")

	# calculate length of fold
	fold_length = int(dataset_size / k)  
	print "Fold length: ", dataset_size, "/ ",k, " = ", fold_length

	average_f1_training_score = []
	average_accuracy_training_score = []
	average_f1_test_score = []
	average_accuracy_test_score = []

	for i in range(k):
		# use one / kth as the test set
		test_set = rows[i * fold_length:(i + 1) * fold_length]
		# use remaining for training
		training_set = rows[:i * fold_length] + rows[(i + 1) * fold_length:]

		# train the model
		classifier.reset()
		if isinstance(classifier, DecisionTree_ID3):
			classifier.train(training_set, classification_label)
		if isinstance(classifier, KNN_Classifier):
			classifier.train(training_set, classification_label, ["citric acid","residual sugar","density"])

		training_score = Classifier_Score();
		test_score = Classifier_Score();

		# evaluate using test set
		for item in training_set:
			ground_truth = item[classification_label]
			# make a prediction using this training set of data
			prediction = classifier.classify(item)
			training_score.record_result(ground_truth, prediction)

		# evaluate using test set
		for item in test_set:
			ground_truth = item[classification_label]
			# make a prediction using this training set of data
			prediction = classifier.classify(item)
			test_score.record_result(ground_truth, prediction)

		print "Fold-%d:" % i
		f1_score = training_score.get_weighted_F1_score()
		accuracy_score = training_score.get_accuracy()
		average_f1_training_score.append(f1_score)
		average_accuracy_training_score.append(accuracy_score)
		print "Training: F1 Score: %.1f , Accuracy: %.1f" % (f1_score * 100.0, accuracy_score * 100.0)
		f1_score = test_score.get_weighted_F1_score()
		accuracy_score = test_score.get_accuracy()
		average_f1_test_score.append(f1_score)
		average_accuracy_test_score.append(accuracy_score)
		print "Test: F1 Score: %.1f , Accuracy: %.1f\n" % (f1_score * 100.0, accuracy_score * 100.0)

	# Print averages
	print "Average:"
	f1_score = numpy.array(average_f1_training_score)
	accuracy_score = numpy.array(average_accuracy_training_score)
	print "Training: F1 Score: %.1f , Accuracy: %.1f" % (numpy.average(f1_score) * 100.0, numpy.average(accuracy_score) * 100.0)
	f1_score = numpy.array(average_f1_test_score)
	accuracy_score = numpy.array(average_accuracy_test_score)
	print "Test: F1 Score: %.1f , Accuracy: %.1f" % (numpy.average(f1_score) * 100.0, numpy.average(accuracy_score) * 100.0)
