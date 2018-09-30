import sys

from Perceptron.datastream import MNIST_Datastream
from Perceptron.digit_classifier import Digit_Classifier_Vanilla

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

	inputs_vector = []
	ground_truth_labels = []

	for i in range(500):
		feature_inputs = training_data_stream.get_rounded_1d_image(i)
		# augment the inputs with the bias term
		feature_inputs.append(1)
		inputs_vector.append(feature_inputs)
		ground_truth_labels.append(training_data_stream.get_label(i))

	vanilla_classifier = Digit_Classifier_Vanilla()
	vanilla_classifier.train(50, inputs_vector, ground_truth_labels, 0.001)

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
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio