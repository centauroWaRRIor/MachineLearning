import math
import collections
import csv
import sys

from KNearestNeighbor.KNN import KNN_Classifier
from DecisionTree.CART import DecisionTree_CART

class DataSet(object):
	def __init__(self):
		# Organizing the data as a list of dictionaries
		# each dictionary contains feature, value for easy
		# retrieval
		self.list_dict = []
		self.classification_label = None

	def load(self, file_name, classification_label):

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

def k_FoldValidation(k, classifier, dataset, classification_label, feature_list, verbose = False):
     
	dataset_size = len(dataset.list_dict)
	if k > dataset_size:
		raise Exception("Bad k-fold argument given size of data")

	# calculate length of fold
	fold_length = int(dataset_size / k)  
	if(verbose):
		print "Fold length: ", dataset_size, "/ ",k, " = ", fold_length

	for i in range(k):
		# use one / kth as the test set
		test_set = dataset.list_dict[i * fold_length:(i + 1) * fold_length]
		# use remaining for training
		training_set = dataset.list_dict[:i * fold_length] + dataset.list_dict[(i + 1) * fold_length:]
		correct_classifications = 0 
		for item in test_set:
			ground_truth = item[classification_label]
			# take a guess using this training set of data
			guess = classifier.classify(item, training_set, feature_list, classification_label)
			if guess == ground_truth:
				correct_classifications += 1
		print "fold %d, correct = %d out of %d" % (i, correct_classifications, len(test_set))
		accuracy = correct_classifications / float(len(test_set))
		print accuracy
		

def main():
	data = DataSet()
	classification_label = "quality"
	data.load("winequality-white.csv", classification_label)

	# simple test
	#classifier = KNN_Classifier()

	item = data.list_dict[101]
	#item_label = item["quality"]
	#print item
	#result = classifier.classify(item, data.list_dict[:100], ["citric acid","residual sugar","density"], data.classification_label)
	#print "result KNN", result

	## k_FoldValidation(4, classifier, data, data.classification_label, ["citric acid","residual sugar","density"], True)

	classifier = DecisionTree_CART()
	#classifier.build(data.list_dict[:100], data.classification_label)
	classifier.build(data.list_dict[:100], data.classification_label)
	classifier.debug_print_tree(classifier.root_node)
	result = classifier.classify(item)

	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio
