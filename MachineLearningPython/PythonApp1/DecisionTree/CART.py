from collections import namedtuple

class DecisionTree_CART(object):

	# Following objects are meant for internal use only

	PartitionDataTuple = namedtuple('PartitionDataTuple', ['true_rows', 'false_rows'])
	BestQuestionInfoTuple = namedtuple('BestQuestionInfoTuple', ['question', 'gain'])

	class Feature_Question:
		"""A Feature_Question is used to partition a dataset.
		This object holds the feature label and its value
		"""

		def __init__(self, column_label, value):
			assert(isinstance(column_label, str))
			assert(isinstance(value, float))
			self.column_label = column_label
			self.value = value

		def answer(self, datapoint):
			# Compare the feature value in a given datapoint to the
			# feature value in this question and reply with true or false
			val = datapoint[self.column_label]
			# Assuming all the values are numeric
			return val >= self.value

		def __repr__(self):
			return "(%s >= %s?)" % (self.column_label, str(self.value))


	class Leaf_Node:
		"""A Leaf node does classification book keeping on the rows that 
		reach it
		"""

		def __init__(self, rows, classification_label):
			self.predictions = DecisionTree_CART.create_class_count_dict(rows, classification_label)

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
		self.root_node = None
		self.classification_label = None
		self.feature_labels = None

	def __feature_unique_values(self, rows, feature):
		"""Find the unique values for a column in a dataset."""

		return set([row[feature] for row in rows])

	@staticmethod
	def create_class_count_dict(rows, classification_label):
		"""Counts the number of each type of label.
		"""

		counts = {}  
		for row in rows:
			label = row[classification_label]
			if label not in counts:
				counts[label] = 0
			counts[label] += 1
		return counts

	def __partition_data(self, rows, question):
		"""Partitions a dataset based on question
		"""
		assert(isinstance(rows, list))
		assert(isinstance(question, self.Feature_Question))
		partition_tuple = self.PartitionDataTuple(true_rows = [], false_rows = [])

		for row in rows:
			assert(isinstance(row, dict))
			if question.answer(row):
				partition_tuple.true_rows.append(row)
			else:
				partition_tuple.false_rows.append(row)
		return partition_tuple

	def __gini(self, rows):
		"""Calculate the Gini Impurity for a list of rows.

		There are a few different ways to do this, I thought this one was
		the most concise. See:
		https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
		"""
		counts = self.create_class_count_dict(rows, self.classification_label)
		impurity = 1
		for lbl in counts:
			prob_of_lbl = counts[lbl] / float(len(rows))
			impurity -= prob_of_lbl**2
		return impurity

	def __info_gain(self, left_rows, right_rows, current_uncertainty):
		"""Information Gain.

	    The uncertainty of the starting node, minus the weighted impurity of
	    two child nodes.
	    """
		p = float(len(left_rows)) / (len(left_rows) + len(right_rows))
		return current_uncertainty - p * self.__gini(left_rows) - (1 - p) * self.__gini(right_rows)

	def __find_best_question(self, rows):
		"""Find the best question to ask by iterating over every feature / value
		and calculating the information gain.
		"""
		best_gain = 0  # keep track of the best information gain
		best_question = None  # keep track of the feature that produced it
		current_uncertainty = self.__gini(rows)
		#total_features = len(self.feature_labels) - 1  # number of columns

		for feature_label in self.feature_labels:

			values = self.__feature_unique_values(rows, feature_label)

			for val in values:  # for each value

				assert(isinstance(val, float)) # only the classification feature is an int
				assert(isinstance(feature_label, str))

				question = self.Feature_Question(feature_label, val)

				partition_tuple = self.__partition_data(rows, question)

				# Skip this split if it doesn't divide the dataset.
				if len(partition_tuple.true_rows) == 0 or len(partition_tuple.false_rows) == 0:
					continue

				# Calculate the information gain from this partition
				gain = self.__info_gain(partition_tuple.true_rows, partition_tuple.false_rows, current_uncertainty)

				if gain > best_gain:
					best_gain = gain
					best_question = question

		return self.BestQuestionInfoTuple(best_question, best_gain)

	def build(self, dataset, classification_label):
		"""Meant to be the external facing build function"""

		self.classification_label = classification_label
		self.feature_labels = dataset[0].keys()
		self.feature_labels.remove(self.classification_label)
		self.root_node = self.__build_tree(dataset)

	def __build_tree(self, rows):
		"""Builds the tree using recursion. Base case is no further
		information gain.
		"""

		# best information gain is associated with asking the most appropiate question
		best_question_info_tuple = self.__find_best_question(rows)

		# no further info gain so base case reached
		if best_question_info_tuple.gain == 0:
			return self.Leaf_Node(rows, self.classification_label)

		# partition data using best question
		partition_tuple = self.__partition_data(rows, best_question_info_tuple.question)

		# true branch recursion Depth First style.
		true_branch = self.__build_tree(partition_tuple.true_rows)

		# false branch recursion Depth First style.
		false_branch = self.__build_tree(partition_tuple.false_rows)

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

		return self.__classify(row, self.root_node)

	def __classify(self, row, node):

		# Base case: we've reached a leaf
		if isinstance(node, self.Leaf_Node):
			return node.predictions

	    # Decide whether to follow the true-branch or the false-branch.
	    # Compare the feature / value stored in the node,
	    # to the example we're considering.
		if node.question.answer(row):
			return self.__classify(row, node.true_branch)
		else:
			return self.__classify(row, node.false_branch)
