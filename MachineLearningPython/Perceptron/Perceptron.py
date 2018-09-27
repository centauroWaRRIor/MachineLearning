
import sys
import struct
import numpy as np


class Online_Datastream:
	"""Simulates an infinite stream of examples"""

	def __init__(self, images_filename, labels_filename):

		# open file
		self.images_file = open(images_filename, 'rb')
		self.labels_file = open(labels_filename, 'rb')

		# extract magic number
		self.images_file.seek(0)
		self.magic_number = struct.unpack('>4B', self.images_file.read(4))
		self.labels_file.seek(0)
		self.magic_number = struct.unpack('>4B', self.labels_file.read(4))

		#32-bit integer (4 bytes)
		self.images_file.seek(4)
		self.num_images = struct.unpack('>I', self.images_file.read(4))[0]
		print "num images: ", self.num_images
		self.num_rows = struct.unpack('>I', self.images_file.read(4))[0]
		print "num rows: ", self.num_rows
		self.num_columns = struct.unpack('>I', self.images_file.read(4))[0]
		print "num columns: ", self.num_columns

		self.labels_file.seek(4)
		assert(self.num_images == struct.unpack('>I', self.labels_file.read(4))[0])

		self.label_bytes = 1 
		self.image_bytes = self.num_rows * self.num_columns # one byte per grid entry


	def __del__(self):

		self.images_file.close()
		self.labels_file.close()

	def get_label(self, image_index):

		file_offset = 8 + image_index * self.label_bytes
		self.labels_file.seek(file_offset) # absolute offset
		label = struct.unpack('>'+'B' * self.label_bytes, self.labels_file.read(self.label_bytes))[0]
		# returns python array
		return label

	def get_image(self, image_index):

		file_offset = 16 + image_index * self.image_bytes
		self.images_file.seek(file_offset) # absolute offset
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * self.image_bytes, self.images_file.read(self.image_bytes))).reshape(self.num_rows, self.num_columns)
		# returns python array
		return image_numpy_array.tolist()


	def ascii_show(self, image):
		for y in image:
			row = ""
			for x in y:
				row += '{0: <4}'.format(x)
			print row

def main():

	images_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-images-idx3-ubyte/train-images.idx3-ubyte"
	labels_filename = "C:/Users/Emmauel/Desktop/CS_578/hw2/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
	data_stream = Online_Datastream(images_filename, labels_filename)
	image_python_array = data_stream.get_image(33)
	data_stream.ascii_show(image_python_array)
	print data_stream.get_label(33)
	
	return 0

if __name__ == "__main__":
	sys.setrecursionlimit(5000)
	#sys.exit(int(main() or 0)) # use for when running without debugging
	main() # use for when debugging within Visual Studio