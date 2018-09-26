
import sys
import struct
import numpy as np

def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row

def main():

	training_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"

	# open file
	training_file = open(training_filename, 'rb')

	# extract magic number
	training_file.seek(0)
	magic_number = struct.unpack('>4B', training_file.read(4))

	#32-bit integer (4 bytes)
	training_file.seek(4)
	num_images = struct.unpack('>I', training_file.read(4))[0]
	print "num images: ", num_images
	num_rows = struct.unpack('>I',training_file.read(4))[0]
	print "num rows: ", num_rows
	num_columns = struct.unpack('>I',training_file.read(4))[0]
	print "num columns: ", num_columns

	image_bytes = num_rows * num_columns

	for i in range(3):
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * image_bytes, training_file.read(image_bytes))).reshape(num_rows, num_columns)
		image_python_array = image_numpy_array.tolist()
		ascii_show(image_python_array)

	training_file.close()
	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio