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
