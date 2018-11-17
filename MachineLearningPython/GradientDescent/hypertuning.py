import sys
import os
import argparse
import gzip
import shutil
import random

from GradientDescent.datastream import MNIST_Datastream
from GradientDescent.digit_classifier import Digit_Classifier

def uncompress_file(abs_path, compress_filename, uncompress_filename):
	with gzip.open(os.path.join(os.path.abspath(abs_path), compress_filename), 'rb') as f_in:
		with open(os.path.join(os.path.abspath(abs_path), uncompress_filename), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)


def test_feature_type_2_data(data_stream):
	"""Sanity check to visually inspect my feature type 2 conversion is correct"""
	image = data_stream.get_image_feature_type_1(1000)
	data_stream.ascii_show(image)
	print("                      ")
	mini_image = data_stream.get_image_feature_type_2(1000)
	data_stream.ascii_show(mini_image)
	print("                      ")
	mini_1d_image = data_stream.get_scaled_1d_image_feature_type_2(1000)
	print mini_1d_image
	print("                      ")
	image = data_stream.get_image_feature_type_1(900)
	data_stream.ascii_show(image)
	print("                      ")
	mini_image = data_stream.get_image_feature_type_2(900)
	data_stream.ascii_show(mini_image)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('regularization', help="Add L2 regularization? [True/False]")
	parser.add_argument('feature_type', help="784 features vs 196 [type1/type2].")
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

	num_features = 0
	
	# collect train data in a random order
	inputs_vector_train = []
	ground_truth_labels_train = []
	random_indices = []
	for i in range(10000):
		random_indices.append(i)
	random.shuffle(random_indices)
	for i in random_indices:
		if args.feature_type == "type1":
			num_features = 28 * 28 + 1 # 785
			feature_inputs = training_data_stream.get_scaled_1d_image_feature_type_1(i)
		elif args.feature_type == "type2":
			num_features = 14 * 14 + 1 # 197
			feature_inputs = training_data_stream.get_scaled_1d_image_feature_type_2(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector_train.append(feature_inputs)
		ground_truth_labels_train.append(training_data_stream.get_label(i))
	
	# collect test data
	inputs_vector_test = []
	ground_truth_labels_test = []
	for i in range(test_data_stream.num_images):
		if args.feature_type == "type1":
			num_features = 28 * 28 + 1 # 785
			feature_inputs = test_data_stream.get_scaled_1d_image_feature_type_1(i)
		elif args.feature_type == "type2":
			num_features = 14 * 14 + 1 # 197
			feature_inputs = test_data_stream.get_scaled_1d_image_feature_type_2(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector_test.append(feature_inputs)
		ground_truth_labels_test.append(test_data_stream.get_label(i))

	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#gd_classifier = Digit_Classifier("Batch", num_features)
	#if args.regularization == "True":
	#	l2_lambda = 0.5 # Found through hyper tuning
	#if args.regularization == "False":
	#	l2_lambda = 0.0 # Found through hyper tuning
	#learning_rate = 0.03 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("convergence.png")
	
	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.001 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_001.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.01 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_01.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.1 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_1.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_10.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.5 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_5.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_type1_l_rate_10.png")

	# this showed that the behavior between type 1 and type 2 does not change with the l_rate
	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.03
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_type1_l_rate_03.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.25
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_10_l2_lambda_25.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.50
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_10_l2_lambda_50.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 0.75
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_10_l2_lambda_75.png")

	#gd_classifier = Digit_Classifier("Batch", num_features)
	#l2_lambda = 1.0
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("hypertuning_l_rate_10_l2_lambda_100.png")
	
	########################## SGD hypertuning ######################################

	#gd_classifier = Digit_Classifier("Stochastic", num_features) 
	#l2_lambda = 0.0
	#learning_rate = 0.001 # crashes
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_001.png")

	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.01 # crashes at 998 epochs with around 0.80 accuracy
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_01.png")
	
	#print "Continue SGD hyper tuning"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_05.png")

	#print "Continue SGD hyper tuning"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_type1_l_rate_05.png")

	#print "Continue SGD hyper tuning"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.1 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_type1_l_rate_1.png")

	#print "Continue SGD hyper tuning"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_type1_l_rate_08.png")

	#print "Continue SGD hyper tuning"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 0.1 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_1.png")

	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.0
	#learning_rate = 1.0 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_10.png")
	################################################################################
	#print "Tune SGD hyper parameter lambda"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.25 # crashed
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_05_l2_lambda_25.png")

	#print "Tune SGD hyper parameter lambda"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.50 crashes
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_05_l2_lambda_50.png")

	#print "Tune SGD hyper parameter lambda"
	#gd_classifier = Digit_Classifier("Stochastic", num_features)
	#l2_lambda = 0.75 crashes
	#learning_rate = 0.05 # Found through hyper tuning
	#gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	#gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_05_l2_lambda_75.png")

	print "Tune SGD hyper parameter lambda"
	gd_classifier = Digit_Classifier("Stochastic", num_features)
	l2_lambda = 0.001 # this could have work with the bug fix, lets try 0.01 next time.
	learning_rate = 0.05 # Found through hyper tuning
	gd_classifier.run_until_convergence(learning_rate, l2_lambda, inputs_vector_train, ground_truth_labels_train, inputs_vector_test, ground_truth_labels_test)
	gd_classifier.plot_train_vs_test_performance("stochastic_hypertuning_l_rate_05_l2_lambda_01.png")

	return 0

if __name__ == "__main__":
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio
