import struct
import numpy as np

class MNIST_Datastream:
	"""Simulates an infinite stream of examples"""

	def __init__(self, images_filename, labels_filename, verbose=False):

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
		self.num_rows = struct.unpack('>I', self.images_file.read(4))[0]		
		self.num_columns = struct.unpack('>I', self.images_file.read(4))[0]
		if verbose:
			print "num images: ", self.num_images
			print "num rows: ", self.num_rows
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


	def get_1d_image(self, image_index):

		file_offset = 16 + image_index * self.image_bytes
		self.images_file.seek(file_offset) # absolute offset
		image_numpy_array = np.asarray(struct.unpack('>'+'B' * self.image_bytes, self.images_file.read(self.image_bytes)))
		# returns python array
		return image_numpy_array.tolist()


	def get_scaled_1d_image(self, image_index):

		image = self.get_1d_image(image_index)
		scaled_image_float = []
		for i in range(self.num_rows * self.num_columns):
			scaled_image_float.append(image[i]/255.0)
		return scaled_image_float

	def get_image_feature_type_2(self, image_index):
	
		def get_largest_neighbor(upper_left, upper_right, botttom_left, bottom_right):
			largest = upper_left
			if largest < upper_right:
				largest = upper_right
			if largest < botttom_left:
				largest = botttom_left
			if largest < bottom_right:
				largest = bottom_right
			return largest
		
		python_2d_array = self.get_image(image_index)
		square_window_size = 2
		python_2d_array_feature_type_2 = []
		for y in range(0, len(python_2d_array), square_window_size):
			python_2d_array_feature_type_2.append([])
			for x in range(0, len(python_2d_array[0]), square_window_size):
				upper_left =  python_2d_array[y][x]
				upper_right = python_2d_array[y][x+1]
				botttom_left = python_2d_array[y+1][x]
				bottom_right = python_2d_array[y+1][x+1]
				largest = get_largest_neighbor(upper_left, upper_right, botttom_left, bottom_right)
				python_2d_array_feature_type_2[-1].append(largest)
		
		return python_2d_array_feature_type_2
		

	def ascii_show(self, image):
		for y in image:
			row = ""
			for x in y:
				row += '{0: <4}'.format(x)
			print row
