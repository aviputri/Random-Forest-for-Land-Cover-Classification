import numpy as np
import arff

#feature class
fc_a = np.load('/Volumes/ga87rif/Study Project/Samples/A_train09.npy')
#digital numbers
dg_a = np.load('/Volumes/ga87rif/Study Project/Samples/A_train_array_09.npy')

#f = np.reshape(feature_class_array, (feature_class_array.shape[0],(feature_class_array.shape[0]-1)))
f = np.reshape(fc_a, (fc_a.shape[0],1))
f = f.astype(np.uint8)

#Stack arrays in sequence horizontally (column wise)
d = np.hstack((dg_a,f))
data = d.tolist()

arff.dump(open('/Volumes/ga87rif/Study Project/Samples/A_train09.arff', 'w'), 
	data, relation="training datasets: case A", 
	names=['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band7', 'NDVI', 'Class'])