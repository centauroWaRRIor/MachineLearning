import sys
import os
import numpy
import matplotlib.pyplot as plt

from Perceptron.datastream import MNIST_Datastream
from Perceptron.digit_classifier import Digit_Classifier

def plot_train_vs_test_performance(x_values, y_values_training, y_values_test, xlabel, ylabel, title):

	plt.figure()
	s = numpy.array(x_values)
	# plot training data
	t = numpy.array(y_values_training)
	plt.plot(s, t, label="Training")
	# plot test data
	t = numpy.array(y_values_test)
	plt.plot(s, t, label="Test")
	plt.legend()

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.savefig(str(title) + ".png")
	#plt.show()

def plot_performance(x_values, y_values, xlabel, ylabel, title):

	plt.figure()
	s = numpy.array(x_values)
	t = numpy.array(y_values)
	plt.plot(s, t)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.savefig(str(title) + ".png")
	#plt.show()

def experiment_training_size(
	classifier_type,
	training_data_stream,
	test_data_stream,
	training_size_initial, 
	training_size_final, 
	training_size_step, 
	number_epoch, 
	learning_rate):

	inputs_vector = []
	ground_truth_labels = []
	plot_x_values = []
	plot_y_train_values = []
	plot_y_test_values = []

	for training_size in range(training_size_initial, training_size_final, training_size_step):
		
		plot_x_values.append(training_size)
		print "Training size = %d" % training_size

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		digit_classifier = Digit_Classifier(classifier_type)
		digit_classifier.train(number_epoch, inputs_vector, ground_truth_labels, learning_rate, True)

		# evaluate performance on the training data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_train_values.append(f1_score)

		# evaluate performance on the test data
		inputs_vector = []
		ground_truth_labels = []
		for i in range(training_size):
			feature_inputs = test_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(test_data_stream.get_label(i))
		# evaluate performance on the test data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_train_vs_test_performance(plot_x_values, plot_y_train_values, plot_y_test_values, "Training size", "F1 Score", "Training_size_effect_on_training_and_test - " + classifier_type)
	plot_performance(plot_x_values, plot_y_train_values, "Training size", "F1 Score", "Training_size_effect_on_training - " + classifier_type)
	plot_performance(plot_x_values, plot_y_test_values, "Training size", "F1 Score", "Training_size_effect_on_test - " + classifier_type)


def experiment_epoch_size(
	classifier_type,
	training_data_stream,
	test_data_stream,
	epoch_size_initial, 
	epoch_size_final, 
	epoch_size_step, 
	training_size, 
	learning_rate):

	inputs_vector = []
	ground_truth_labels = []
	plot_x_values = []
	plot_y_train_values = []
	plot_y_test_values = []

	for epoch_size in range(epoch_size_initial, epoch_size_final, epoch_size_step):
		
		plot_x_values.append(epoch_size)
		print "Epoch size = %d" % epoch_size

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		digit_classifier = Digit_Classifier(classifier_type)
		digit_classifier.train(epoch_size, inputs_vector, ground_truth_labels, learning_rate, True)

		# evaluate performance on the training data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_train_values.append(f1_score)

		# evaluate performance on the test data
		inputs_vector = []
		ground_truth_labels = []
		for i in range(training_size):
			feature_inputs = test_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(test_data_stream.get_label(i))
		# evaluate performance on the test data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_train_vs_test_performance(plot_x_values, plot_y_train_values, plot_y_test_values, "Epoch size", "F1 Score", "Epoch_size_effect_on_training_and_test - " + classifier_type)
	plot_performance(plot_x_values, plot_y_train_values, "Epoch size", "F1 Score", "Epoch_size_effect_on_training - " + classifier_type)
	plot_performance(plot_x_values, plot_y_test_values, "Epoch size", "F1 Score", "Epoch_size_effect_on_test - " + classifier_type)


def experiment_learning_rates(
	classifier_type,
	training_data_stream,
	test_data_stream,
	learning_rates, 
	training_size,
	number_epoch):

	inputs_vector = []
	ground_truth_labels = []
	plot_x_values = []
	plot_y_train_values = []
	plot_y_test_values = []

	for learning_rate in learning_rates:
		
		plot_x_values.append(learning_rate)
		print "Learning Rate = %f" % learning_rate

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		digit_classifier = Digit_Classifier(classifier_type)
		digit_classifier.train(number_epoch, inputs_vector, ground_truth_labels, learning_rate, True)

		# evaluate performance on the training data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_train_values.append(f1_score)

		# evaluate performance on the test data
		inputs_vector = []
		ground_truth_labels = []
		for i in range(training_size):
			feature_inputs = test_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(test_data_stream.get_label(i))
		# evaluate performance on the test data
		f1_score = digit_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_train_vs_test_performance(plot_x_values, plot_y_train_values, plot_y_test_values, "Learning rate", "F1 Score", "Learning_rate_effect_on_training_and_test - " + classifier_type)
	plot_performance(plot_x_values, plot_y_train_values, "Learning rate", "F1 Score", "Learning_rate_effect_on_training - " + classifier_type)
	plot_performance(plot_x_values, plot_y_test_values, "Learning rate", "F1 Score", "Learning_rate_effect_on_test - " + classifier_type)


def main():

	#data_path = "C:\Users\Emmauel\Desktop\CS_578\hw2\data"
	data_path = "C:\cs_homework"
	images_filename_path = os.path.join(os.path.abspath(data_path), "train-images.idx3-ubyte")
	labels_filename_path = os.path.join(os.path.abspath(data_path), "train-labels.idx1-ubyte")
	training_data_stream = MNIST_Datastream(images_filename_path, labels_filename_path, True)

	images_filename_path = os.path.join(os.path.abspath(data_path), "t10k-images.idx3-ubyte")
	labels_filename_path = os.path.join(os.path.abspath(data_path), "t10k-labels.idx1-ubyte")
	test_data_stream = MNIST_Datastream(images_filename_path, labels_filename_path, True)

	#image_python_array = training_data_stream.get_image(33)
	#training_data_stream.ascii_show(image_python_array)
	#print training_data_stream.get_label(33)
	#print "--------------"
	#image_python_array = training_data_stream.get_rounded_image(33)
	#training_data_stream.ascii_show(image_python_array)
	
	#image_python_array = training_data_stream.get_rounded_1d_image(33)
	#print image_python_array
	
	#f1_scoring = Classifier_Score()
	#y_true = [0, 1, 2, 0, 1, 2]
	#y_pred = [0, 2, 1, 0, 0, 1]
	#for true, pred in zip(y_true, y_pred):
	#	f1_scoring.record_result(true, pred)
	#print "F1-Score: ", f1_scoring.get_macro_F1_score()

	# Vanilla perceptron hyper parameters search
	# completed experiment_training_size("Vanilla_Perceptron", training_data_stream, test_data_stream, 500, 10000, 250, 50, 0.001)
	# completed experiment_epoch_size("Vanilla_Perceptron", training_data_stream, test_data_stream, 10, 100, 5, 10000, 0.001)
	#learning_rates = [0.0001, 0.001, 0.01, 0.1]
	#experiment_learning_rates("Vanilla_Perceptron", training_data_stream, test_data_stream, learning_rates, 10000, 50)

	# Average perceptron hyper parameters search
	#experiment_training_size("Average_Perceptron", training_data_stream, test_data_stream, 500, 10000, 250, 50, 0.001)
	experiment_epoch_size("Average_Perceptron", training_data_stream, test_data_stream, 10, 100, 5, 10000, 0.001)
	learning_rates = [0.0001, 0.001, 0.01, 0.1]
	experiment_learning_rates("Average_Perceptron", training_data_stream, test_data_stream, learning_rates, 10000, 50)

	# Winnow hyper parameters search
	experiment_training_size("Winnow_Perceptron", training_data_stream, test_data_stream, 500, 10000, 250, 50, 0.001)
	experiment_epoch_size("Winnow_Perceptron", training_data_stream, test_data_stream, 10, 100, 5, 10000, 0.001)
	# I did not implement winnow with learning rates factor so no point in tuning this parameter

	# small experiment for debugging this code
	#experiment_training_size("Average_Perceptron", training_data_stream, test_data_stream, 500, 600, 25, 25, 0.01)
	#experiment_epoch_size("Average_Perceptron", training_data_stream, test_data_stream, 10, 100, 25, 500, 0.001)
	#learning_rates = [0.0001, 0.001, 0.01, 0.1]
	#experiment_learning_rates("Average_Perceptron", training_data_stream, test_data_stream, learning_rates, 500, 25)

	# TODO: Shuffle the inputs

	return 0

if __name__ == "__main__":
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio