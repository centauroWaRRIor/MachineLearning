import sys

from Classifier.KNN import KNN_Classifier
from Classifier.ID3 import DecisionTree_ID3
from Classifier.dataset import DataSet
from Classifier.dataset import k_FoldValidation
from Classifier.dataset import k_FoldValidation_ID3_tuning

def main():
	data = DataSet()
	classification_label = "quality"
	data.load("winequality-white.csv", classification_label)

	# simple test
	#classifier = KNN_Classifier()

	#item = data.list_dict[101]
	#item_label = item["quality"]
	#print item
	#result = classifier.classify(item, data.list_dict[:100], ["citric acid","residual sugar","density"], data.classification_label)
	#print "result KNN", result

	classifier = DecisionTree_ID3()
	#classifier.build(data.list_dict[:100], data.classification_label)
	#classifier.build(data.list_dict[20:80], data.classification_label)
	#classifier.debug_print_tree(classifier.root_node)
	#result = classifier.classify(item)
	#print "result Decision Tree", result

	
	#k_FoldValidation(4, classifier, data.list_dict[:800], classification_label)
	k_FoldValidation_ID3_tuning(4, classifier, data.list_dict[:800], classification_label)

	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio
