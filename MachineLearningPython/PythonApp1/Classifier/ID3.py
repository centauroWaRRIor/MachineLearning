
import math
import numpy
from collections import namedtuple

class DecisionTree_ID3(object):

	# Following objects are meant for internal use only

	PartitionDataTuple = namedtuple('PartitionDataTuple', ['true_rows', 'false_rows'])
	BestQuestionInfoTuple = namedtuple('BestQuestionInfoTuple', ['question', 'gain'])

	class Split_Question:
		"""A Split_Question is used to partition a dataset.
		This object holds the feature label and its value
		"""

		def __init__(self, feature_label, value):
			assert(isinstance(feature_label, str))
			assert(isinstance(value, float))
			self.feature_label = feature_label
			self.value = value

		def answer(self, datapoint):
			# Compare the feature value in a given datapoint to the
			# feature value in this question and reply with true or false
			val = datapoint[self.feature_label]
			# Assuming all the values are numeric
			return val >= self.value

		def __repr__(self):
			return "(%s >= %s?)" % (self.feature_label, str(self.value))


	class Leaf_Node:
		"""A Leaf node does classification book keeping on the rows that 
		reach it. Ideally, all the rows (examples) in it have the same label
		"""

		def __init__(self, rows, classification_label):
			self.predictions = DecisionTree_ID3.create_count_dict(rows, classification_label)

		def __repr__(self):
			"""Print the predictions at a leaf as a dictionary format."""
			total = sum(self.predictions.values()) * 1.0
			probs = {}
			for classification_label in self.predictions.keys():
				probs[classification_label] = str(int(self.predictions[classification_label] / total * 100)) + "%"
			return str(probs)


	class Decision_Node:
		"""A Decision Node is used to ask a question during classification 
		It takes care of the question and child nodes book keeping.
		"""

		def __init__(self, question, true_branch, false_branch):
			self.question = question
			self.true_branch = true_branch
			self.false_branch = false_branch


	def __init__(self):
		reset()

	def reset(self):
		self.root_node = None
		self.classification_label = None
		self.feature_labels = None

	@staticmethod
	def create_count_dict(rows, column):
		"""Counts the number of each type of label.
		"""

		counts = {}  
		for row in rows:
			label = row[column]
			if label not in counts:
				counts[label] = 0
			counts[label] += 1
		return counts

	def __partition_data(self, rows, question):
		"""Partitions a dataset based on question
		"""
		assert(isinstance(rows, list))
		assert(isinstance(question, self.Split_Question))
		partition_tuple = self.PartitionDataTuple(true_rows = [], false_rows = [])

		for row in rows:
			assert(isinstance(row, dict))
			if question.answer(row):
				partition_tuple.true_rows.append(row)
			else:
				partition_tuple.false_rows.append(row)
		return partition_tuple

	def __entropy(self, rows):
		"""Calculates current entropy of the set of examples
		"""
		counts_dict = self.create_count_dict(rows, self.classification_label)

		dataEntropy = 0.0
		for key in counts_dict.keys():
			freq = counts_dict[key]
			p = freq/float(len(rows))
			dataEntropy += -p * math.log(p, 2)
		
		return dataEntropy


	def __expected_entropy(self, target_attribute_values_array):
		"""Calculates expected entropy of current set of examples
		    with respect to a certain target attribute
		"""

		dataEntropy = 0.0
		
		total_rows = 0
		for target_attribute_values_set in target_attribute_values_array:
			total_rows += len(target_attribute_values_set)

		for target_attribute_values_set in target_attribute_values_array:
			freq = len(target_attribute_values_set)
			dataEntropy += (freq/total_rows) * self.__entropy(target_attribute_values_set)

		return dataEntropy

	def __info_gain(self, true_rows, false_rows, entropy):
		"""Information Gain.

	    Calculates the expected reduction of entropy caused by partitioning on
		candidate attribute.
	    """
		attribute_values_array = []
		attribute_values_array.append(true_rows)
		attribute_values_array.append(false_rows)
		return entropy - self.__expected_entropy(attribute_values_array)


	def __find_best_split(self, rows, attribute_labels):
		"""Find the best question to ask by iterating over every feature / value
		and calculating the information gain.
		"""
		best_gain = 0  # keep track of the best information gain
		best_question = None  # keep track of the feature that produced it

		for feature_label in attribute_labels:

			current_entropy = self.__entropy(rows)
			assert(isinstance(feature_label, str))
			question = self.Split_Question(feature_label, self.value_attribute_threshold[feature_label])
			partition_tuple = self.__partition_data(rows, question)

			# Skip this split if it doesn't divide the dataset as it will not reduce the entropy
			if len(partition_tuple.true_rows) == 0 or len(partition_tuple.false_rows) == 0:
				continue

			# Calculate the information gain from this partition
			gain = self.__info_gain(partition_tuple.true_rows, partition_tuple.false_rows, current_entropy)

			if gain > best_gain:
				best_gain = gain
				best_question = question

		return self.BestQuestionInfoTuple(best_question, best_gain)


	def train(self, dataset, classification_label):
		"""Meant to be the external facing train function"""

		self.classification_label = classification_label
		self.feature_labels = dataset[0].keys()
		self.feature_labels.remove(self.classification_label)

		# build the thresholds once
		self.value_attribute_threshold = {}
		for attribute in self.feature_labels:
			temp_set = []
			for row in dataset:
				temp_set.append(row[attribute])
			numpy_array = numpy.array(temp_set)
			self.value_attribute_threshold[attribute] = numpy.mean(numpy_array)

		self.root_node = self.__build_tree(dataset, self.feature_labels[:])

	def __build_tree(self, rows, feature_labels):
		"""Builds the tree using recursion. Base case is no further
		information gain.
		"""

		# best information gain is associated with asking the most appropiate question
		best_question_info_tuple = self.__find_best_split(rows, feature_labels[:])

		# no further info gain so base case reached
		if best_question_info_tuple.gain == 0:
			return self.Leaf_Node(rows, self.classification_label)

		# partition data using best question
		partition_tuple = self.__partition_data(rows, best_question_info_tuple.question)

		# remove this feature from the feature list since it has been used to split
		feature_labels.remove(best_question_info_tuple.question.feature_label)

		# true branch recursion Depth First style.
		true_branch = self.__build_tree(partition_tuple.true_rows, feature_labels[:]) # note the pass by value instead of pass by reference, very important

		# false branch recursion Depth First style.
		false_branch = self.__build_tree(partition_tuple.false_rows, feature_labels[:]) # note the pass by value instead of pass by reference, very important

		# decision node will hold the best question to ask at this point
		# as well as it will hold the two branches
		return self.Decision_Node(best_question_info_tuple.question, true_branch, false_branch)

	def debug_print_tree(self, node, spacing=" "):

		# Base case: we've reached a leaf
		if isinstance(node, self.Leaf_Node):
			print spacing + "Predictions", node
			return

		# Print the question at this node
		print spacing + str(node.question)

		# Call this function recursively on the true branch
		print spacing + 'left_branch:'
		self.debug_print_tree(node.true_branch, spacing + "  ")

		# Call this function recursively on the false branch
		print spacing + 'right_branch:'
		self.debug_print_tree(node.false_branch, spacing + "  ")

	def classify(self, row):
		"""Meant to be the public function for classifying"""

		leaf_node = self.__classify(row, self.root_node)
		prediction = self.__majority_vote(leaf_node)
		return prediction
		
	def __majority_vote(self, leaf_node):

		majority_vote = 0
		majority_label = None
		for label in leaf_node.predictions.keys():
			if leaf_node.predictions[label] > majority_vote:
				majority_label = label
				majority_vote = leaf_node.predictions[label]
		return majority_label

	def __classify(self, row, node):

		# Base case: we've reached a leaf
		if isinstance(node, self.Leaf_Node):
			return node

	    # Decide whether to follow the true-branch or the false-branch.
	    # Compare the feature / value stored in the node,
	    # to the example we're considering.
		if node.question.answer(row):
			return self.__classify(row, node.true_branch)
		else:
			return self.__classify(row, node.false_branch)
