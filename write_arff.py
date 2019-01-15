"""

data = [[1,2,'a'], [3, 4, 'john']]
arff.dump(open('result.arff', 'w'), data, relation="whatever", names=['num', 'day', 'title'])

Result:

@relation whatever
@attribute num integer
@attribute day integer
@attribute title string
@data
1,2,'a'
3,4,'john'

"""

import arff
import numpy as np

def writearff(feature_class_array, digital_number_array, filename, relation_name, field_id):

	"""
	e.g.
	filename = '/dir/abc.arff'
	relation_name = 'satellite image'
	field_id = ['b1','b2','b3','b4','b5','b7','ndvi','class']

	"""

	#reshape the feature_class_array from (n,) to (n,1)
	#f = np.reshape(feature_class_array, (feature_class_array.shape[0],(feature_class_array.shape[0]-1)))
	f = np.reshape(feature_class_array, (feature_class_array.shape[0],1))

	#Stack arrays in sequence horizontally (column wise)
	data = np.hstack((digital_number_array,f))

	arff.dump(open(filename, 'w'), data, relation=relation_name, names=field_id)