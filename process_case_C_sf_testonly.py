import functions
import numpy as np

"""
locate the Numpy array inputs
"""

#training datasets
#class
train09 = np.load('Samples/SLC off/C_train09.npy')
#surface reflectance value
train_array_09 = np.load('Samples/SLC off/C_train_array_sf_09.npy')

#class
train11 = np.load('Samples/SLC off/C_train11.npy')
#surface reflectance value
train_array_11 = np.load('Samples/SLC off/C_train_array_sf_11.npy')

#ground truth datasets for validation
test09 = np.load('Samples/SLC off/C_test09.npy')
test11 = np.load('Samples/SLC off/C_test11.npy')

"""
locate the satellite images
"""
#sat image 2009
b1_09 = 'L2 imagery/2009/clip_b1r.tif'
b2_09 = 'L2 imagery/2009/clip_b2r.tif'
b3_09 = 'L2 imagery/2009/clip_b3r.tif'
b4_09 = 'L2 imagery/2009/clip_b4r.tif'
b5_09 = 'L2 imagery/2009/clip_b5r.tif'
b7_09 = 'L2 imagery/2009/clip_b7r.tif'
ndvi_09 = 'L2 imagery/2009/ndvi.tif'
ndwi_09 = 'L2 imagery/2009/ndwi.tif'
mndwi1_09 = 'L2 imagery/2009/mndwi1.tif'
mndwi2_09 = 'L2 imagery/2009/mndwi2.tif'
ndbi_09 = 'L2 imagery/2009/ndbi.tif'
mndbi_09 = 'L2 imagery/2009/mndbi.tif'

#sat image 2011
b1_11 = 'L2 imagery/2011/clip_b1r.tif'
b2_11 = 'L2 imagery/2011/clip_b2r.tif'
b3_11 = 'L2 imagery/2011/clip_b3r.tif'
b4_11 = 'L2 imagery/2011/clip_b4r.tif'
b5_11 = 'L2 imagery/2011/clip_b5r.tif'
b7_11 = 'L2 imagery/2011/clip_b7r.tif'
ndvi_11 = 'L2 imagery/2011/ndvi.tif'
ndwi_11 = 'L2 imagery/2011/ndwi.tif'
mndwi1_11 = 'L2 imagery/2011/mndwi1.tif'
mndwi2_11 = 'L2 imagery/2011/mndwi2.tif'
ndbi_11 = 'L2 imagery/2011/ndbi.tif'
mndbi_11 = 'L2 imagery/2011/mndbi.tif'

"""
locate the test shapefiles
"""
file_train09_shp = 'Samples/SLC off/2percent/s09train.shp'
file_train11_shp = 'Samples/SLC off/2percent/s11train.shp'
file_test09_shp = 'Samples/SLC off/2percent/s09test.shp'
file_test11_shp = 'Samples/SLC off/2percent/s11test.shp'

#-----------------------------------------------------------------------------------------------
#read raster values and combine them into train_array (test datasets)

#2009
b1_09_t = functions.extract_values(shp = file_test09_shp, raster = b1_09)
b2_09_t = functions.extract_values(shp = file_test09_shp, raster = b2_09)
b3_09_t = functions.extract_values(shp = file_test09_shp, raster = b3_09)
b4_09_t = functions.extract_values(shp = file_test09_shp, raster = b4_09)
b5_09_t = functions.extract_values(shp = file_test09_shp, raster = b5_09)
b7_09_t = functions.extract_values(shp = file_test09_shp, raster = b7_09)
ndvi_09_t = functions.extract_values(shp = file_test09_shp, raster = ndvi_09)
ndwi_09_t = functions.extract_values(shp = file_test09_shp, raster = ndwi_09)
mndwi1_09_t = functions.extract_values(shp = file_test09_shp, raster = mndwi1_09)
mndwi2_09_t = functions.extract_values(shp = file_test09_shp, raster = mndwi2_09)
ndbi_09_t = functions.extract_values(shp = file_test09_shp, raster = ndbi_09)
mndbi_09_t = functions.extract_values(shp = file_test09_shp, raster = mndbi_09)

test_09 = functions.combine_bands_sf(b1 = b1_09_t, b2 = b2_09_t, b3 = b3_09_t, b4 = b4_09_t, 
	b5 = b5_09_t, b7 = b7_09_t, ndvi = ndvi_09_t, ndwi = ndwi_09_t, mndwi1 = mndwi1_09_t, 
	mndwi2 = mndwi2_09_t, ndbi = ndbi_09_t, mndbi = mndbi_09_t,
	multiband_array_file = 'Samples/SLC off/C_test_array_sf_09.npy')

#2011
b1_11_t = functions.extract_values(shp = file_test11_shp, raster = b1_11)
b2_11_t = functions.extract_values(shp = file_test11_shp, raster = b2_11)
b3_11_t = functions.extract_values(shp = file_test11_shp, raster = b3_11)
b4_11_t = functions.extract_values(shp = file_test11_shp, raster = b4_11)
b5_11_t = functions.extract_values(shp = file_test11_shp, raster = b5_11)
b7_11_t = functions.extract_values(shp = file_test11_shp, raster = b7_11)
ndvi_11_t = functions.extract_values(shp = file_test11_shp, raster = ndvi_11)
ndwi_11_t = functions.extract_values(shp = file_test11_shp, raster = ndwi_11)
mndwi1_11_t = functions.extract_values(shp = file_test11_shp, raster = mndwi1_11)
mndwi2_11_t = functions.extract_values(shp = file_test11_shp, raster = mndwi2_11)
ndbi_11_t = functions.extract_values(shp = file_test11_shp, raster = ndbi_11)
mndbi_11_t = functions.extract_values(shp = file_test11_shp, raster = mndbi_11)

test_11 = functions.combine_bands_sf(b1 = b1_11_t, b2 = b2_11_t, b3 = b3_11_t, b4 = b4_11_t, 
	b5 = b5_11_t, b7 = b7_11_t, ndvi = ndvi_11_t, ndwi = ndwi_11_t, mndwi1 = mndwi1_11_t, 
	mndwi2 = mndwi2_11_t, ndbi = ndbi_11_t, mndbi = mndbi_11_t,
	multiband_array_file = 'Samples/SLC off/C_test_array_sf_11.npy')

#-----------------------------------------------------------------------------------------------
#train and predict with RF

#2009
#100 trees
test_result_09_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_100trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_09_100.npy')
# None == n_features

#200 trees
test_result_09_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_200trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_09_200.npy')
# None == n_features

#300 trees
test_result_09_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_300trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_09_300.npy')
# None == n_features

#400 trees
test_result_09_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_400trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_09_400.npy')
# None == n_features

#500 trees
test_result_09_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_500trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_09_500.npy')
# None == n_features

#2011
#100 trees
test_result_11_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_100trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_11_100.npy')
# None == n_features

#200 trees
test_result_11_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_200trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_11_200.npy')
# None == n_features

#300 trees
test_result_11_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_300trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_11_300.npy')
# None == n_features

#400 trees
test_result_11_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_400trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_11_400.npy')
# None == n_features

#500 trees
test_result_11_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_500trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_sf_11_500.npy')
# None == n_features


#-----------------------------------------------------------------------------------------------
# calculate accuracy

print("Case C (n(samples) = 2% n(raster pixels)")
print(" ")
functions.test_accuracy(year = 2009, trees = 100, test_array = test_result_09_100, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 200, test_array = test_result_09_200, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 300, test_array = test_result_09_300, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 400, test_array = test_result_09_400, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 500, test_array = test_result_09_500, gt_test_array = test09)

functions.test_accuracy(year = 2011, trees = 100, test_array = test_result_11_100, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 200, test_array = test_result_11_200, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 300, test_array = test_result_11_300, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 400, test_array = test_result_11_400, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 500, test_array = test_result_11_500, gt_test_array = test11)
