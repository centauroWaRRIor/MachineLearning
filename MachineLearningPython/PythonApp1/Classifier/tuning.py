import matplotlib.pyplot as plt
import numpy
from Classifier.KNN import KNN_Classifier
from Classifier.ID3 import DecisionTree_ID3
from Classifier.dataset import Classifier_Score

def k_FoldValidation_ID3_tuning(k, rows, classification_label):
     
	ID3_classifier = DecisionTree_ID3()
	dataset_size = len(rows)
	if k > dataset_size:
		raise Exception("Bad k-fold argument given size of data")

	# calculate length of fold
	fold_length = int(dataset_size / k)  
	print "Fold length: ", dataset_size, "/ ",k, " = ", fold_length

	hyper_parameter_array = [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12]
	hyper_parameter_result_arrays = []

	for i in range(k):
		
		f1_scores = []

		# use one / kth as the test set
		test_set = rows[i * fold_length:(i + 1) * fold_length]
		# use remaining for training
		training_set = rows[:i * fold_length] + rows[(i + 1) * fold_length:]
		# among training set, reserve 20% for validation tuning
		validation_set_size = int(0.2 * len(training_set))
		validation_set = training_set[:validation_set_size]
		# resize the training set
		training_set = training_set[validation_set_size:]

		for hyper_parameter in hyper_parameter_array:
			# train the model
			ID3_classifier.reset()
			ID3_classifier.max_depth = hyper_parameter
			ID3_classifier.train(training_set, classification_label)

			validation_score = Classifier_Score();

			# evaluate using validation set
			for item in validation_set:
				ground_truth = item[classification_label]
				# make a prediction using this training set of data
				prediction = ID3_classifier.classify(item)
				validation_score.record_result(ground_truth, prediction)

			print "Fold-%d:" % (i+1)
			print "Hyper parameter max_depth = %d" % hyper_parameter
			f1_score = validation_score.get_macro_F1_score()
			accuracy_score = validation_score.get_accuracy()
			print "Validation: F1 Score: %.1f , Accuracy: %.1f" % (f1_score * 100.0, accuracy_score * 100.0)
			f1_scores.append(f1_score)

		hyper_parameter_result_arrays.append(f1_scores)
	plot_performance(hyper_parameter_array, hyper_parameter_result_arrays, 'Max Depth', 'F1 Score', 'ID3 Max Depth Tuning')


def k_FoldValidation_KNN_tuning(k, rows, classification_label, distance_function="Euclidean"):

	KNN_classifier = KNN_Classifier()
	dataset_size = len(rows)
	if k > dataset_size:
		raise Exception("Bad k-fold argument given size of data")

	# calculate length of fold
	fold_length = int(dataset_size / k)  
	print "Fold length: ", dataset_size, "/ ",k, " = ", fold_length

	hyper_parameter_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
	hyper_parameter_result_arrays = []

	for i in range(k):
		
		f1_scores = []

		# use one / kth as the test set
		test_set = rows[i * fold_length:(i + 1) * fold_length]
		# use remaining for training
		training_set = rows[:i * fold_length] + rows[(i + 1) * fold_length:]
		# among training set, reserve 20% for validation tuning
		validation_set_size = int(0.2 * len(training_set))
		validation_set = training_set[:validation_set_size]
		# resize the training set
		training_set = training_set[validation_set_size:]

		for hyper_parameter in hyper_parameter_array:
			# train the model
			KNN_classifier.reset()
			KNN_classifier.k_neighbors = hyper_parameter
			if distance_function is "Euclidean":
				KNN_classifier.distance_function = KNN_classifier.euclidean_distance
			elif distance_function is "Cosine_Similarity":
				KNN_classifier.distance_function = KNN_classifier.cosine_simmilarity
			dimensions = ["citric acid","residual sugar","density"]
			KNN_classifier.train(training_set, classification_label, dimensions)

			validation_score = Classifier_Score();

			# evaluate using validation set
			for item in validation_set:
				ground_truth = item[classification_label]
				# make a prediction using this training set of data
				prediction = KNN_classifier.classify(item)
				validation_score.record_result(ground_truth, prediction)

			print "Fold-%d:" % (i+1)
			print "Hyper parameter k-nieghbor = %d" % hyper_parameter
			f1_score = validation_score.get_macro_F1_score()
			accuracy_score = validation_score.get_accuracy()
			print "Validation: F1 Score: %.1f , Accuracy: %.1f" % (f1_score * 100.0, accuracy_score * 100.0)
			f1_scores.append(f1_score)

		hyper_parameter_result_arrays.append(f1_scores)
	plot_performance(hyper_parameter_array, hyper_parameter_result_arrays, 'K Neighbor', 'F1 Score', 'KNN K-neighbor Tuning (%s)' % distance_function)


def plot_performance(x_values, y_values_arrays, xlabel, ylabel, title):

	for fold_array in y_values_arrays:
		s = numpy.array(x_values)
		t = numpy.array(fold_array)
		plt.plot(s, t)


	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	#plt.savefig("test.png")
	plt.show()
