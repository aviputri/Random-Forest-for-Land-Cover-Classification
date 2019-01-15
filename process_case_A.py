import functions
import numpy as np
import time

#Case A: n(sample points) == 0.1% * n(raster pixels)

#locate all needed files
file_train09_dbf = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s09train.dbf'
file_test09_dbf = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s09test.dbf'

file_train11_dbf = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s11train.dbf'
file_test11_dbf = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s11test.dbf'

file_train09_shp = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s09train.shp'
file_train11_shp = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s11train.shp'
file_test09_shp = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s09test.shp'
file_test11_shp = '/Volumes/ga87rif/Study Project/Samples/0.1percent/s11test.shp'

b1_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b1r.tif'
b2_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b2r.tif'
b3_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b3r.tif'
b4_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b4r.tif'
b5_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b5r.tif'
b7_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/clip_b7r.tif'
ndvi_09 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2009/ndvi.tif'

b1_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b1r.tif'
b2_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b2r.tif'
b3_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b3r.tif'
b4_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b4r.tif'
b5_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b5r.tif'
b7_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/clip_b7r.tif'
ndvi_11 = '/Volumes/ga87rif/Study Project/satelite images/Level-2/2011/ndvi.tif'

#-----------------------------------------------------------------------------------------------
# read all train and test sample points

#2009
#0.1 percent sample data
#train
train09 = functions.read_class(tabular_data=file_train09_dbf, 
	gt_array_file='/Volumes/ga87rif/Study Project/Samples/A_train09.npy')
#test
test09 = functions.read_class(tabular_data=file_test09_dbf,
	gt_array_file='/Volumes/ga87rif/Study Project/Samples/A_test09.npy')

#2011
#0.1 percent sample data
#train
train11 = functions.read_class(tabular_data=file_train11_dbf, 
	gt_array_file='/Volumes/ga87rif/Study Project/Samples/A_train11.npy')
#test
test11 = functions.read_class(tabular_data=file_test11_dbf,
	gt_array_file='/Volumes/ga87rif/Study Project/Samples/A_test11.npy')

#-----------------------------------------------------------------------------------------------
#read raster values and combine them into train_array

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
	multiband_array_file = '/Volumes/ga87rif/Study Project/Samples/A_train_array_09.npy')

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
	multiband_array_file = '/Volumes/ga87rif/Study Project/Samples/A_train_array_11.npy')

#-----------------------------------------------------------------------------------------------
#read image bands and combine them into img_array

#2009
img_b1_09 = functions.create_band(band = b1_09)
img_b2_09 = functions.create_band(band = b2_09)
img_b3_09 = functions.create_band(band = b3_09)
img_b4_09 = functions.create_band(band = b4_09)
img_b5_09 = functions.create_band(band = b5_09)
img_b7_09 = functions.create_band(band = b7_09)
img_ndvi_09 = functions.create_band(band = ndvi_09)

img_09 = functions.combine_bands(b1 = img_b1_09, b2 = img_b2_09, b3 = img_b3_09, b4 = img_b4_09, 
	b5 = img_b5_09, b7 = img_b7_09, ndvi = img_ndvi_09, 
	multiband_array_file = '/Volumes/ga87rif/Study Project/satelite images/Level-2/img_09.npy')

#2011
img_b1_11 = functions.create_band(band = b1_11)
img_b2_11 = functions.create_band(band = b2_11)
img_b3_11 = functions.create_band(band = b3_11)
img_b4_11 = functions.create_band(band = b4_11)
img_b5_11 = functions.create_band(band = b5_11)
img_b7_11 = functions.create_band(band = b7_11)
img_ndvi_11 = functions.create_band(band = ndvi_11)

img_11 = functions.combine_bands(b1 = img_b1_11, b2 = img_b2_11, b3 = img_b3_11, b4 = img_b4_11, 
	b5 = img_b5_11, b7 = img_b7_11, ndvi = img_ndvi_11, 
	multiband_array_file = '/Volumes/ga87rif/Study Project/satelite images/Level-2/img_11.npy')

#-----------------------------------------------------------------------------------------------
#train and predict with RF

#2009
#100 trees
result_09_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_09_100trees.sav', 
	img = img_09,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.npy')
# None == n_features

#200 trees
result_09_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_09_200trees.sav', 
	img = img_09,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_200.npy')

#300 trees
result_09_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_09_300trees.sav', 
	img = img_09,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_300.npy')

#400 trees
result_09_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_09_400trees.sav', 
	img = img_09,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_400.npy')

#500 trees
result_09_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_09_500trees.sav', 
	img = img_09,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_500.npy')


#2011
#100 trees
result_11_100 = functions.train_rf(trees = 100, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_11_100trees.sav', 
	img = img_11,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_100.npy')

#200 trees
result_11_200 = functions.train_rf(trees = 200, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_11_200trees.sav', 
	img = img_11,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_200.npy')

#300 trees
result_11_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_11_300trees.sav', 
	img = img_11,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_300.npy')

#400 trees
result_11_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_11_400trees.sav', 
	img = img_11,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_400.npy')

#500 trees
result_11_500 = functions.train_rf(trees = 500, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = '/Users/aviputripertiwi/Documents/TU Munchen/Study Project/A_rf_11_500trees.sav', 
	img = img_11,
	result_array_file = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_500.npy')

#-----------------------------------------------------------------------------------------------
# make rasters from results
#2009
functions.rasterize(result_array = result_09_100, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
functions.rasterize(result_array = result_09_200, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_200.tif')
functions.rasterize(result_array = result_09_300, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_300.tif')
functions.rasterize(result_array = result_09_400, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_400.tif')
functions.rasterize(result_array = result_09_500, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_500.tif')

#2011
functions.rasterize(result_array = result_11_100, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_100.tif')
functions.rasterize(result_array = result_11_200, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_200.tif')
functions.rasterize(result_array = result_11_300, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_300.tif')
functions.rasterize(result_array = result_11_400, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_400.tif')
functions.rasterize(result_array = result_11_500, 
	result_raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_11_500.tif')

#-----------------------------------------------------------------------------------------------
# extract result values to test points

#2009
test_result_09_100 = functions.extract_values(shp = file_test09_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_09_100.npy', test_result_09_100)
test_result_09_200 = functions.extract_values(shp = file_test09_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_09_200.npy', test_result_09_200)
test_result_09_300 = functions.extract_values(shp = file_test09_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_09_300.npy', test_result_09_300)
test_result_09_400 = functions.extract_values(shp = file_test09_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_09_400.npy', test_result_09_400)
test_result_09_500 = functions.extract_values(shp = file_test09_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_09_500.npy', test_result_09_500)

#2011
test_result_11_100 = functions.extract_values(shp = file_test11_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_11_100.npy', test_result_11_100)
test_result_11_200 = functions.extract_values(shp = file_test11_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_11_200.npy', test_result_11_200)
test_result_11_300 = functions.extract_values(shp = file_test11_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_11_300.npy', test_result_11_300)
test_result_11_400 = functions.extract_values(shp = file_test11_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_11_400.npy', test_result_11_400)
test_result_11_500 = functions.extract_values(shp = file_test11_shp, 
	raster = '/Volumes/ga87rif/Study Project/Result/New/A_result_09_100.tif')
np.save('/Volumes/ga87rif/Study Project/Samples/A_test_result_11_500.npy', test_result_11_500)

#-----------------------------------------------------------------------------------------------
# calculate accuracy

functions.test_accuracy(year = 2009, trees = 100, test_array = test_result_09_100, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 200, test_array = test_result_09_100, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 300, test_array = test_result_09_100, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 400, test_array = test_result_09_100, gt_test_array = test09)
functions.test_accuracy(year = 2009, trees = 500, test_array = test_result_09_100, gt_test_array = test09)

functions.test_accuracy(year = 2011, trees = 100, test_array = test_result_11_100, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 200, test_array = test_result_11_100, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 300, test_array = test_result_11_100, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 400, test_array = test_result_11_100, gt_test_array = test11)
functions.test_accuracy(year = 2011, trees = 500, test_array = test_result_11_100, gt_test_array = test11)



#-----------------------------------------------------------------------------------------------
# Calculate time

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))