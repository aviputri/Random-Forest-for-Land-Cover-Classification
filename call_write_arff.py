import write_arff
import numpy as np

#load the numpy array files
"""
case A
"""
#feature class
fc_a = np.load('/Volumes/ga87rif/Study Project/Samples/A_train09.npy')
#digital numbers
dg_a = np.load('/Volumes/ga87rif/Study Project/Samples/A_train_array_09.npy')

"""
case B
"""
#feature class
fc_b = np.load('/Volumes/ga87rif/Study Project/Samples/B_train09.npy')
#digital numbers
dg_b = np.load('/Volumes/ga87rif/Study Project/Samples/B_train_array_09.npy')

"""
case C
"""
#feature class
fc_c = np.load('/Volumes/ga87rif/Study Project/Samples/C_train09.npy')
#digital numbers
dg_c = np.load('/Volumes/ga87rif/Study Project/Samples/C_train_array_09.npy')

#write case A
write_arff.writearff(feature_class_array = fc_a, digital_number_array = dg_a, 
	filename = '/Volumes/ga87rif/Study Project/Samples/A_train09.arff', 
	relation_name = 'training datasets: case A', 
	field_id = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 7', 'NDVI', 'Land Cover Class'])

#write case B
write_arff.writearff(feature_class_array = fc_b, digital_number_array = dg_b, 
	filename = '/Volumes/ga87rif/Study Project/Samples/B_train09.arff', 
	relation_name = 'training datasets: case B', 
	field_id = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 7', 'NDVI', 'Land Cover Class'])

#write case C
write_arff.writearff(feature_class_array = fc_c, digital_number_array = dg_c, 
	filename = '/Volumes/ga87rif/Study Project/Samples/C_train09.arff', 
	relation_name = 'training datasets: case C', 
	field_id = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 7', 'NDVI', 'Land Cover Class'])
