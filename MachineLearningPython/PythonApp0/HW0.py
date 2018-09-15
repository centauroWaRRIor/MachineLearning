
import csv
import sys
import re
from Queue import Queue

class Graph_Node(object):

	def __init__(self, node_id, value):
		self.node_id = node_id
		self.value = value
		self.visited = False
		self.adj_id_list = []


class BFS_Graph_MinElement(object):
	""" BFS using recursion.
	"""

	def __init__(self, file_name):

		self.graph_dict = {}
		self.min_value = sys.maxint

		with open(file_name, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				node_1_index = str(0)
				node_1_value = str(0)
				node_2_index = str(0)
				node_2_value = str(0)
				match = re.search(r'<(\d+):(\d+)>', line[0])
				if match:
					node_1_index = match.group(1)
					node_1_value = match.group(2)
				else:
					raise Exception("Unknown formatting")
				match = re.search(r'<(\d+):(\d+)>', line[1])
				if match:
					node_2_index = match.group(1)
					node_2_value = match.group(2)
				else:
					raise Exception("Unknown formatting")
				#print node_1_index, node_1_value,":",node_2_index, node_2_value
				# fill in dict
				if node_1_index in self.graph_dict:
					list = self.graph_dict[node_1_index].adj_id_list
					list.append(node_2_index)
				else: 
					list = []
					list.append(node_2_index)
					graph_node = Graph_Node(node_1_index, node_1_value)
					graph_node.adj_id_list = list
					self.graph_dict[node_1_index] = graph_node
				if node_2_index in self.graph_dict:
					list = self.graph_dict[node_2_index].adj_id_list
					list.append(node_1_index)
				else: 
					list = []
					list.append(node_1_index)
					graph_node = Graph_Node(node_2_index, node_2_value)
					graph_node.adj_id_list = list
					self.graph_dict[node_2_index] = graph_node


	def visit_node(self, node_id):
		#print node_id,":",self.graph_dict[node_id].value
		if int(self.graph_dict[node_id].value) < self.min_value:
			self.min_value = self.graph_dict[node_id].value

	def BFS_recursive(self, aux_queue):
		if aux_queue.empty():
			return

		# visit the node
		node_id = aux_queue.get()
		self.visit_node(node_id)
		
		# add all adjacent nodes to the to visit 
		for adj_node in self.graph_dict[node_id].adj_id_list:
			if not self.graph_dict[adj_node].visited:
				self.graph_dict[adj_node].visited = True # considered visited now
				aux_queue.put(self.graph_dict[adj_node].node_id)
		
		self.BFS_recursive(aux_queue)

	def SearchMinValue(self):
		aux_queue = Queue()
		for node_id in self.graph_dict: # traversing keys
			if not self.graph_dict[node_id].visited:
				self.graph_dict[node_id].visited = True # considered visited now
				aux_queue.put(node_id)
				self.BFS_recursive(aux_queue)
		print 'Min value: ', self.min_value
		

def main():
	test = BFS_Graph_MinElement('nodes.csv')
	test.SearchMinValue()
	return 0


if __name__ == "__main__":
	sys.setrecursionlimit(5000) # homework asked for recursion implementation explicitly
	sys.exit(int(main() or 0)) # use for when running without debugging
	#main() # use for when debugging 
