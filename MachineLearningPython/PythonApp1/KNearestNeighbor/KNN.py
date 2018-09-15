from collections import namedtuple

class KNN_Classifier(object):

	DatapointTuple = namedtuple('DatapointTuple', ['datapoint', 'distance'])

	def __init__(self):

		self.k_neighbors = 3 # hyper parameter

	def __euclidean_distance(self, data_point1, data_point2, feature_list):

		dimension = len(feature_list)
		distance = 0
		for x in range(dimension):
			value1 = data_point1[feature_list[x]]
			value2 = data_point2[feature_list[x]]
			distance += pow((value2 - value1), 2)
		return math.sqrt(distance)

	def classify(self, item_classify, dataset, feature_list, classification_label):

		closest_distance_list = []
		# iterate through all the data points and build list of closest neighbors
		for datapoint in dataset:
			# compute euclidean distance
			distance = self.__euclidean_distance(item_classify, datapoint, feature_list)
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
		return self.__majority_vote_label(closest_distance_list, classification_label)


	def __majority_vote_label(self, closest_distance_list, classification_label):

		count_dict = {}
		# cast the votes
		for item in closest_distance_list:
			classification = item.datapoint[classification_label]
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

