import math
from collections import namedtuple

class KNN_Classifier(object):

	DatapointTuple = namedtuple('DatapointTuple', ['datapoint', 'distance'])

	def __init__(self):
		self.reset()

	def reset(self):
		self.distance_features_list = []
		self.k_neighbors = 0 # hyper parameter
		self.classification_label = None
		self.feature_labels = []
		self.dataset = []

	def train(self, dataset, classification_label, distance_features_list):
		self.distance_features_list = distance_features_list
		self.k_neighbors = len(self.distance_features_list)
		self.dataset = dataset # this classifier memorizes more than learn
		self.classification_label = classification_label
		self.feature_labels = dataset[0].keys()
		self.feature_labels.remove(self.classification_label)

	def __euclidean_distance(self, data_point1, data_point2):

		dimension = len(self.distance_features_list)
		distance = 0
		for x in range(dimension):
			value1 = data_point1[self.distance_features_list[x]]
			value2 = data_point2[self.distance_features_list[x]]
			distance += pow((value2 - value1), 2)
		return math.sqrt(distance)

	def classify(self, item_classify):

		closest_distance_list = []
		# iterate through all the data points and build list of closest neighbors
		for datapoint in self.dataset:
			# compute euclidean distance
			distance = self.__euclidean_distance(item_classify, datapoint)
			distance_item = self.DatapointTuple(datapoint, distance)
			# pass through add items until closest distance is initially filled up
			if len(closest_distance_list) < self.k_neighbors:
				closest_distance_list.append(distance_item)
				closest_distance_list.sort(key=lambda x: x.distance)
			# compare against largest distance found so far and insert if new found distance is less
			elif distance < closest_distance_list[self.k_neighbors-1].distance: 
				closest_distance_list.pop() # remove old largest value
				closest_distance_list.append(distance_item)
				closest_distance_list.sort(key=lambda x: x.distance)
		#print "Closest distance list", closest_distance_list
		return self.__majority_vote_label(closest_distance_list)


	def __majority_vote_label(self, closest_distance_list, ):

		count_dict = {}
		# cast the votes
		for item in closest_distance_list:
			classification = item.datapoint[self.classification_label]
			if classification in count_dict.keys():
				count = count_dict[classification]
				count_dict[classification] = count + 1
			else:
				count_dict[classification] = 1
		# count the votes
		winner_label = 0
		largest_count = -1
		for key in count_dict.keys():
			if count_dict[key] > largest_count:
				largest_count = count_dict[key]
				winner_label = key
		return winner_label

