import sys

from Classifier.baseline import Random_Classifier
from Classifier.dataset import DataSet
from Classifier.dataset import k_FoldValidation

def main():
	data = DataSet()
	classification_label = "quality"
	data.load("winequality-white.csv", classification_label, False)

	classifier = Random_Classifier()
	k_FoldValidation(4, classifier, data.list_dict, classification_label)
	return 0

if __name__ == "__main__":
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio
