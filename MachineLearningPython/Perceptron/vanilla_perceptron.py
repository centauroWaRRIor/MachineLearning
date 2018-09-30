import sys
import os
import argparse

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('training_set_size', required=True, help="Size of training set.")
	parser.add_argument('number_epochs', required=True, help="Number of epochs.")
	parser.add_argument('learning_rate', required=True, help="Learning rate.")
	parser.add_argument('data_path', required=True, help="Path to data folder.")
	args = parser.parse_args(argv)
	
	images_filename_path = os.path.join(os.path.abspath(args.data_path), "train-images.idx3-ubyte")
	labels_filename_path = os.path.join(os.path.abspath(args.data_path), "train-labels.idx1-ubyte")
	training_data_stream = MNIST_Datastream(images_filename_path, labels_filename_path)

	images_filename_path = os.path.join(os.path.abspath(args.data_path), "t10k-images.idx3-ubyte")
	labels_filename_path = os.path.join(os.path.abspath(args.data_path), "t10k-labels.idx1-ubyte")
	test_data_stream = MNIST_Datastream(images_filename_path, labels_filename_path)

	inputs_vector = []
	ground_truth_labels = []

	for i in range(int(args.training_set_size)):
		feature_inputs = training_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(training_data_stream.get_label(i))

	vanilla_classifier = Digit_Classifier_Vanilla()
	vanilla_classifier.train(int(args.number_epochs), inputs_vector, ground_truth_labels, float(args.learning_rate))

	# Predict the training data
	print "Predict the training data: "
	vanilla_classifier.predict(inputs_vector, ground_truth_labels)

	print "Predict the test data: "
	inputs_vector = []
	ground_truth_labels = []
	for i in range(500):
		feature_inputs = test_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(test_data_stream.get_label(i))
	vanilla_classifier.predict(inputs_vector, ground_truth_labels)

	return 0

if __name__ == "__main__":
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio