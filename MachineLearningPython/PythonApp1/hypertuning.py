import sys

from Classifier.dataset import DataSet
from Classifier.tuning import k_FoldValidation_ID3_tuning
from Classifier.tuning import k_FoldValidation_KNN_tuning

def main():
	data = DataSet()
	classification_label = "quality"
	data.load("winequality-white.csv", classification_label, False)
	data.normalize_dataset()

	# tune decision tree hyper parameter
	#k_FoldValidation_ID3_tuning(4, data.list_dict, classification_label)

	# tune KNN hyper parameters
	#k_FoldValidation_KNN_tuning(4, data.list_dict, classification_label, "Euclidean")
	k_FoldValidation_KNN_tuning(4, data.list_dict, classification_label, "Cosine_Similarity")

	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	sys.exit(int(main() or 0)) # use for when running without debugging
	#main() # use for when debugging within Visual Studio
