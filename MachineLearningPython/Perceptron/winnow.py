import sys
import os
import argparse
import gzip
import shutil

from Perceptron.datastream import MNIST_Datastream
from Perceptron.digit_classifier import Digit_Classifier

def uncompress_file(abs_path, compress_filename, uncompress_filename):
	with gzip.open(os.path.join(os.path.abspath(abs_path), compress_filename), 'rb') as f_in:
		with open(os.path.join(os.path.abspath(abs_path), uncompress_filename), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('training_set_size', help="Size of training set.")
	parser.add_argument('number_epochs', help="Number of epochs.")
	parser.add_argument('learning_rate', help="Learning rate.")
	parser.add_argument('data_path', help="Path to data folder.")
	argv = sys.argv[1:]
	args = parser.parse_args(argv)

	# extract gzip files
	uncompress_file(args.data_path, "train-images-idx3-ubyte.gz", "train-images.idx3-ubyte")
	uncompress_file(args.data_path, "train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte")
	uncompress_file(args.data_path, "t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte")
	uncompress_file(args.data_path, "t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")
	
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

	vanilla_classifier = Digit_Classifier("Winnow_Perceptron")
	vanilla_classifier.train(int(args.number_epochs), inputs_vector, ground_truth_labels, float(args.learning_rate))

	# Predict the training data
	f1_macro_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
	print "Training F1 Score: %.2f" % f1_macro_score

	# Predict the test data
	inputs_vector = []
	ground_truth_labels = []
	for i in range(test_data_stream.num_images):
		feature_inputs = test_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(test_data_stream.get_label(i))
	f1_macro_score = vanilla_classifier.evaluate_f1_performance(inputs_vector, ground_truth_labels)
	print "Test F1 Score: %.2f" % f1_macro_score

	return 0

if __name__ == "__main__":
	sys.exit(int(main() or 0)) # use for when running without debugging
	#main() # use for when debugging within Visual Studio

