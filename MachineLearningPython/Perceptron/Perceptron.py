
import sys
import struct
import numpy as np


class Online_Datastream:
	"""Simulates an infinite stream of examples"""

	def __init__(self, training_filename):

		# open file
		self.training_file = open(training_filename, 'rb')

		# extract magic number
		self.training_file.seek(0)
		self.magic_number = struct.unpack('>4B', self.training_file.read(4))

		#32-bit integer (4 bytes)
		self.training_file.seek(4)
		self.num_images = struct.unpack('>I', self.training_file.read(4))[0]
		print "num images: ", self.num_images
		self.num_rows = struct.unpack('>I', self.training_file.read(4))[0]
		print "num rows: ", self.num_rows
		self.num_columns = struct.unpack('>I', self.training_file.read(4))[0]
		print "num columns: ", self.num_columns

		self.image_bytes = self.num_rows * self.num_columns


	def __del__(self):

		self.training_file.close()


	def show_image(self, image_index):

		file_offset = 16 + image_index * self.image_bytes
		self.training_file.seek(file_offset)
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * self.image_bytes, self.training_file.read(self.image_bytes))).reshape(self.num_rows, self.num_columns)
		image_python_array = image_numpy_array.tolist()
		self.ascii_show(image_python_array)


	def ascii_show(self, image):
		for y in image:
			row = ""
			for x in y:
				row += '{0: <4}'.format(x)
			print row

def main():

	training_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	#label_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	data_stream = Online_Datastream(training_filename)
	data_stream.show_image(1)
	
	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio