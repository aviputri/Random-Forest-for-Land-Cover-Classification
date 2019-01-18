import pandas as pd
from simpledbf import Dbf5 #use to read DBF files
from osgeo import gdal, gdal_array, ogr
import struct
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib #use to save the model as .SAV
from sklearn.metrics import accuracy_score, confusion_matrix
import rasterio
from rasterio.transform import from_origin

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

"""
If there are changes in the library, re-import the library with:
import imp
imp.reload(main)
"""

#read the class of each data sample
def read_class(tabular_data, gt_array_file):

	"""
	give an $id column (I scall it FID) to all your DBF file so you can call it in order when
	matching the feature class with the respective Reflectance (in extract_values).
	I gave the $id column in QGIS so there is no code line for that here.
	"""

	dbf = Dbf5(tabular_data) # tabular_data = '/location/test.dbf'
	#convert DBF into Pandas dataframe
	df = dbf.to_dataframe()
	#get Feature ID (FID) and Class ID (Id)
	class_id = df[['FID', 'Id']] #[] == __getitem__ syntax
	#convert from Pandas dataframe into Numpy array
	a = class_id.values
	#sort based on FID (usually this is already sorted but better make it sure here)
	a = a[a[:,0].argsort(kind='mergesort')]
	#take out only the Class ID
	b = a[:,1]
	#convert into unsigned integer datatype
	b = b.astype(np.uint8)
	#make sure the Class ID array is flat
	gt_array = b.ravel()

	# save into .npy
	np.save(gt_array_file, gt_array) #gt_array save = '/location/array.npy'

	#output = gt_array
	return(gt_array)

#extract raster values of each sample point and save it into numpy array
# this extract 1 band
def extract_values(shp, raster):

	"""
	extract the FID with the Reflectance, sort it based on FID, then take only the Reflectance
	"""

	#open raster
	src_ds = gdal.Open(raster)
	gt = src_ds.GetGeoTransform()
	rb = src_ds.GetRasterBand(1)

	#open shapefile
	ds = ogr.Open(shp)
	lyr=ds.GetLayer()
	li_values = list()
	for feat in lyr:
		geom = feat.GetGeometryRef()
		feat_id = feat.GetField('FID')
		mx,my=geom.GetX(), geom.GetY()  #coord in map units

		#Convert from map to pixel coordinates
		#Only works for geotransforms with no rotation
		px = int((mx - gt[0]) / gt[1]) #x pixel
		py = int((my - gt[3]) / gt[5]) #y pixel

		intval=rb.ReadAsArray(px,py,1,1)
		li_values.append([feat_id, intval[0]])
		#convert into numpy array
		a = np.array(li_values).astype(np.float64)
		#sort by FID
		a = a[a[:,0].argsort(kind='mergesort')]
		#take out only the class id
		b = a[:,1]
    	b = b.ravel()
	
	band_array = b.astype(np.float64)

	return(band_array)

#combine the bands
def combine_bands(b1, b2, b3, b4, b5, b7, ndvi, multiband_array_file):

	b1 = b1.ravel()
	b2 = b2.ravel()
	b3 = b3.ravel()
	b4 = b4.ravel()
	b5 = b5.ravel()
	b7 = b7.ravel()
	ndvi = ndvi.ravel()
	comb = np.array([b1, b2, b3, b4, b5, b7, ndvi])
	multiband_array = comb.transpose()

	np.save(multiband_array_file, multiband_array)

	return(multiband_array)

#combine the bands with SF
def combine_bands_sf(b1, b2, b3, b4, b5, b7, ndvi, ndwi, mndwi1, mndwi2, ndbi, mndbi, 
	multiband_array_file):

	b1 = b1.ravel()
	b2 = b2.ravel()
	b3 = b3.ravel()
	b4 = b4.ravel()
	b5 = b5.ravel()
	b7 = b7.ravel()
	ndvi = ndvi.ravel()
	ndwi = ndwi.ravel()
	mndwi1 = mndwi1.ravel()
	mndwi2 = mndwi2.ravel()
	ndbi = ndbi.ravel()
	mndbi = mndbi.ravel()
	comb = np.array([b1, b2, b3, b4, b5, b7, ndvi, ndwi, mndwi1, mndwi2, ndbi, mndbi])
	multiband_array = comb.transpose()

	np.save(multiband_array_file, multiband_array)

	return(multiband_array)

#import the image band to be predicted (one by one)
def create_band(band):

	band_ds = gdal.Open(band, gdal.GA_ReadOnly)
	band_array = band_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)

	return(band_array)

	#then call combine_bands!!

#train the random forest and predict images
def train_rf(trees, maxfeatures, train_array, gt_array, model_sav, img, 
	result_array_file):

	rf = RandomForestClassifier(n_estimators = trees, min_samples_split = 10, 
		max_features = maxfeatures, oob_score=False)
	rf = rf.fit(train_array, gt_array)
	#save model
	#joblib.dump(rf, model_sav)

	result_array = rf.predict(img)
	np.save(result_array_file, result_array)

	return(result_array)


#make raster file from the result
def rasterize(result_array, result_raster):

	a = result_array.astype(np.uint8)

	#reshape into 2D
	b = np.reshape(a, (-1,3057))
	c = np.flipud(b) #flip vertically
	#because rasterio writes from bottom to top

	#transform = from_origin(110....<--dari origin, -7....<--bukan dari origin, 0.000271658, -0.000271658)
	transform = from_origin(110.0363020639564269, -7.8701315292535803, 0.000271658, -0.000271658)

	new_dataset = rasterio.open(result_raster, 'w', driver='GTiff',
		height = b.shape[0], width = c.shape[1],
		count=1, dtype=c.dtype,
		crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
		transform=transform)

	new_dataset.write(c, 1)
	new_dataset.close()

	#then call the extract_values function!!
	#output: test_array

def test_accuracy(year, trees, test_array, gt_test_array):

	a = accuracy_score(gt_test_array, test_array)
	b = confusion_matrix(gt_test_array, test_array, labels=[1,2,3,4,5,6,7,8])

	print(year)
	print 'The overall accuracy of ',trees,' trees is: ',a
	print(b)

