import sys
import numpy
import matplotlib.pyplot as plt

from Perceptron.datastream import MNIST_Datastream
from Perceptron.digit_classifier import Digit_Classifier_Vanilla

def plot_performance(x_values, y_values_training, y_values_test, xlabel, ylabel, title):

	s = numpy.array(x_values)
	# plot training data
	t = numpy.array(y_values_training)
	plt.plot(s, t)
	# plot test data
	t = numpy.array(y_values_test)
	plt.plot(s, t)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	#plt.savefig(str(title) + ".png")
	plt.show()

def experiment_training_size(
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
		print ("Training size = ", training_size)

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		vanilla_classifier = Digit_Classifier_Vanilla()
		vanilla_classifier.train(number_epoch, inputs_vector, ground_truth_labels, learning_rate)

		# evaluate performance on the training data
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
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
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_performance(plot_x_values, plot_y_train_values, "Training size", "F1 Score", "Training_size_effect_on_training")
	plot_performance(plot_x_values, plot_y_test_values, "Training size", "F1 Score", "Training_size_effect_on_test")


def experiment_epoch_size(
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
		print ("Epoch size = ", epoch_size)

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		vanilla_classifier = Digit_Classifier_Vanilla()
		vanilla_classifier.train(epoch_size, inputs_vector, ground_truth_labels, learning_rate)

		# evaluate performance on the training data
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
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
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_performance(plot_x_values, plot_y_train_values, "Epoch size", "F1 Score", "Epoch_size_effect_on_training")
	plot_performance(plot_x_values, plot_y_test_values, "Epoch size", "F1 Score", "Epoch_size_effect_on_test")

def experiment_learning_rates(
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
		print ("Learning Rate = ", learning_rate)

		# build the training data stream
		for i in range(training_size):
			feature_inputs = training_data_stream.get_rounded_1d_image(i)
			# augment the inputs with the bias term
			feature_inputs.append(1)
			inputs_vector.append(feature_inputs)
			ground_truth_labels.append(training_data_stream.get_label(i))

		# train the classifier using the data stream above
		vanilla_classifier = Digit_Classifier_Vanilla()
		vanilla_classifier.train(number_epoch, inputs_vector, ground_truth_labels, learning_rate)

		# evaluate performance on the training data
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
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
		f1_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
		plot_y_test_values.append(f1_score)

	plot_performance(plot_x_values, plot_y_train_values, "Learning rate", "F1 Score", "Learning_rate_effect_on_training")
	plot_performance(plot_x_values, plot_y_test_values, "Learning rate", "F1 Score", "Learning_rate_effect_on_test")


def main():

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
	training_data_stream = MNIST_Datastream(images_filename, labels_filename)

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
	test_data_stream = MNIST_Datastream(images_filename, labels_filename)

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

	#experiment_training_size(training_data_stream, test_data_stream, 500, 10000, 250, 50, 0.001)
	#experiment_epoch_size(training_data_stream, test_data_stream, 10, 100, 5, 10000, 0.001)
	#learning_rates[0.0001, 0.001, 0.01, 0.1]
	#experiment_learning_rates(training_data_stream, test_data_stream, learning_rates, 10000, 50)


	x_values = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750]
	y_values = [
		1.0, 
		1.0, 
		0.9444647883464968,
		0.9316369860846253,
		0.9539819117306605,
		0.9462842348006936,
		0.9347440113963981,
		0.9177641558432594,
		0.865770341259597,
		0.9077938566700892,
		0.9058248131312314,
		0.882906906475057,
		0.8903238895295373,
		0.8711095387587084,
		0.8936526311471003,
		0.8412500580692163,
		0.8872244709293726,
		0.8340285424254892,
		0.833242091908402,
		0.8593284475929812,
		0.8680233970804304,
		0.8653950331083673,
		0.8482893380101142,
		0.8122259194678711,
		0.8286552996833084,
		0.888214510266204,
		0.839798865576564,
		0.863517231989,
		0.8497689931097149,
		0.8451532891146056,
		0.8171746642354382,
		0.8843901724226825,
		0.7969492784953041,
		0.8442400253336606,
		0.8452473415820834,
		0.8637872896686408,
		0.8626166462949275,
		0.8351358408897447]

	y_values_2 = list(y_values)
	for i in range(len(y_values)):
		y_values_2[i] = y_values_2[i] - 0.5

	plot_performance(x_values, y_values, y_values_2, "Epoch size", "F1 Score", "Epoch_size_effect_on_test")

	return 0

if __name__ == "__main__":
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio