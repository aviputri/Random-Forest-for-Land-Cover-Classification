import functions
import numpy as np
import time

#Case C: n(sample points) == 2% * n(raster pixels)

# Start the timer
start_time = time.time()
#-----------------------------------------------------------------------------------------------

#locate all needed files
#DBF files (tabular data of Shapefiles)
file_train09_dbf = 'Samples/SLC off/2percent/s09train.dbf'
file_test09_dbf = 'Samples/SLC off/2percent/s09test.dbf'
file_train11_dbf = 'Samples/SLC off/2percent/s11train.dbf'
file_test11_dbf = 'Samples/SLC off/2percent/s11test.dbf'

#ESRI shapefiles
file_train09_shp = 'Samples/SLC off/2percent/s09train.shp'
file_train11_shp = 'Samples/SLC off/2percent/s11train.shp'
file_test09_shp = 'Samples/SLC off/2percent/s09test.shp'
file_test11_shp = 'Samples/SLC off/2percent/s11test.shp'

#sat image 2009
b1_09 = 'L2 imagery/2009/clip_b1r.tif'
b2_09 = 'L2 imagery/2009/clip_b2r.tif'
b3_09 = 'L2 imagery/2009/clip_b3r.tif'
b4_09 = 'L2 imagery/2009/clip_b4r.tif'
b5_09 = 'L2 imagery/2009/clip_b5r.tif'
b7_09 = 'L2 imagery/2009/clip_b7r.tif'
ndvi_09 = 'L2 imagery/2009/ndvi.tif'

#sat image 2011
b1_11 = 'L2 imagery/2011/clip_b1r.tif'
b2_11 = 'L2 imagery/2011/clip_b2r.tif'
b3_11 = 'L2 imagery/2011/clip_b3r.tif'
b4_11 = 'L2 imagery/2011/clip_b4r.tif'
b5_11 = 'L2 imagery/2011/clip_b5r.tif'
b7_11 = 'L2 imagery/2011/clip_b7r.tif'
ndvi_11 = 'L2 imagery/2011/ndvi.tif'

#-----------------------------------------------------------------------------------------------
# read all train and test sample points

#2009
#0.1 percent sample data
#train
train09 = functions.read_class(tabular_data=file_train09_dbf, 
	gt_array_file='Samples/SLC off/C_train09.npy')
#test
test09 = functions.read_class(tabular_data=file_test09_dbf,
	gt_array_file='Samples/SLC off/C_test09.npy')

#2011
#0.1 percent sample data
#train
train11 = functions.read_class(tabular_data=file_train11_dbf, 
	gt_array_file='Samples/SLC off/C_train11.npy')
#test
test11 = functions.read_class(tabular_data=file_test11_dbf,
	gt_array_file='Samples/SLC off/C_test11.npy')

#-----------------------------------------------------------------------------------------------
#read raster values and combine them into train_array (training datasets)

#2009
b1_09_ar = functions.extract_values(shp = file_train09_shp, raster = b1_09)
b2_09_ar = functions.extract_values(shp = file_train09_shp, raster = b2_09)
b3_09_ar = functions.extract_values(shp = file_train09_shp, raster = b3_09)
b4_09_ar = functions.extract_values(shp = file_train09_shp, raster = b4_09)
b5_09_ar = functions.extract_values(shp = file_train09_shp, raster = b5_09)
b7_09_ar = functions.extract_values(shp = file_train09_shp, raster = b7_09)
ndvi_09_ar = functions.extract_values(shp = file_train09_shp, raster = ndvi_09)

train_array_09 = functions.combine_bands(b1 = b1_09_ar, b2 = b2_09_ar, b3 = b3_09_ar, b4 = b4_09_ar, 
	b5 = b5_09_ar, b7 = b7_09_ar, ndvi = ndvi_09_ar,
	multiband_array_file = 'Samples/SLC off/C_train_array_09.npy')

#2011
b1_11_ar = functions.extract_values(shp = file_train11_shp, raster = b1_11)
b2_11_ar = functions.extract_values(shp = file_train11_shp, raster = b2_11)
b3_11_ar = functions.extract_values(shp = file_train11_shp, raster = b3_11)
b4_11_ar = functions.extract_values(shp = file_train11_shp, raster = b4_11)
b5_11_ar = functions.extract_values(shp = file_train11_shp, raster = b5_11)
b7_11_ar = functions.extract_values(shp = file_train11_shp, raster = b7_11)
ndvi_11_ar = functions.extract_values(shp = file_train11_shp, raster = ndvi_11)

train_array_11 = functions.combine_bands(b1 = b1_11_ar, b2 = b2_11_ar, b3 = b3_11_ar, b4 = b4_11_ar, 
	b5 = b5_11_ar, b7 = b7_11_ar, ndvi = ndvi_11_ar,
	multiband_array_file = 'Samples/SLC off/C_train_array_11.npy')

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

test_09 = functions.combine_bands(b1 = b1_09_t, b2 = b2_09_t, b3 = b3_09_t, b4 = b4_09_t, 
	b5 = b5_09_t, b7 = b7_09_t, ndvi = ndvi_09_t,
	multiband_array_file = 'Samples/SLC off/C_test_array_09.npy')

#2011
b1_11_t = functions.extract_values(shp = file_test11_shp, raster = b1_11)
b2_11_t = functions.extract_values(shp = file_test11_shp, raster = b2_11)
b3_11_t = functions.extract_values(shp = file_test11_shp, raster = b3_11)
b4_11_t = functions.extract_values(shp = file_test11_shp, raster = b4_11)
b5_11_t = functions.extract_values(shp = file_test11_shp, raster = b5_11)
b7_11_t = functions.extract_values(shp = file_test11_shp, raster = b7_11)
ndvi_11_t = functions.extract_values(shp = file_test11_shp, raster = ndvi_11)

test_11 = functions.combine_bands(b1 = b1_11_t, b2 = b2_11_t, b3 = b3_11_t, b4 = b4_11_t, 
	b5 = b5_11_t, b7 = b7_11_t, ndvi = ndvi_11_t,
	multiband_array_file = 'Samples/SLC off/C_test_array_11.npy')

#-----------------------------------------------------------------------------------------------
#train and predict with RF

#2009
#100 trees
test_result_09_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_09_100trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_09_100.npy')
# None == n_features

#200 trees
test_result_09_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_09_200trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_09_200.npy')

#300 trees
test_result_09_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_09_300trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_09_300.npy')

#400 trees
test_result_09_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_09_400trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_09_400.npy')

#500 trees
test_result_09_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_09_500trees.sav', 
	img = test_09,
	result_array_file = 'L2 imagery/Results/C_test_result_09_500.npy')


#2011
#100 trees
test_result_11_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_11_100trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_11_100.npy')

#200 trees
test_result_11_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_11_200trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_11_200.npy')

#300 trees
test_result_11_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_11_300trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_11_300.npy')

#400 trees
test_result_11_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_11_400trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_11_400.npy')

#500 trees
test_result_11_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_11_500trees.sav', 
	img = test_11,
	result_array_file = 'L2 imagery/Results/C_test_result_11_500.npy')

#-----------------------------------------------------------------------------------------------
# calculate accuracy

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



#-----------------------------------------------------------------------------------------------
# Print the timer

print("--- %s seconds ---" % (time.time() - start_time))