import functions
import numpy as np

#locate all needed files
#DBF files (tabular data of Shapefiles)
file_train09_dbf = 'Samples/SLC off/1percent/s09train.dbf'
file_test09_dbf = 'Samples/SLC off/1percent/s09test.dbf'
file_train11_dbf = 'Samples/SLC off/1percent/s11train.dbf'
file_test11_dbf = 'Samples/SLC off/1percent/s11test.dbf'

#ESRI shapefiles
file_train09_shp = 'Samples/SLC off/1percent/s09train.shp'
file_train11_shp = 'Samples/SLC off/1percent/s11train.shp'
file_test09_shp = 'Samples/SLC off/1percent/s09test.shp'
file_test11_shp = 'Samples/SLC off/1percent/s11test.shp'

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

#-----------------------------------------------------------------------------------------------
# read all train and test sample points

#2009
#0.1 percent sample data
#train
train09 = functions.read_class(tabular_data=file_train09_dbf, 
	gt_array_file='Samples/SLC off/B_train09.npy')
#test
test09 = functions.read_class(tabular_data=file_test09_dbf,
	gt_array_file='Samples/SLC off/B_test09.npy')

#2011
#0.1 percent sample data
#train
train11 = functions.read_class(tabular_data=file_train11_dbf, 
	gt_array_file='Samples/SLC off/B_train11.npy')
#test
test11 = functions.read_class(tabular_data=file_test11_dbf,
	gt_array_file='Samples/SLC off/B_test11.npy')

#-----------------------------------------------------------------------------------------------
##read raster values and combine them into train_array (training datasets)

#2009
b1_09_ar = functions.extract_values(shp = file_train09_shp, raster = b1_09)
b2_09_ar = functions.extract_values(shp = file_train09_shp, raster = b2_09)
b3_09_ar = functions.extract_values(shp = file_train09_shp, raster = b3_09)
b4_09_ar = functions.extract_values(shp = file_train09_shp, raster = b4_09)
b5_09_ar = functions.extract_values(shp = file_train09_shp, raster = b5_09)
b7_09_ar = functions.extract_values(shp = file_train09_shp, raster = b7_09)
ndvi_09_ar = functions.extract_values(shp = file_train09_shp, raster = ndvi_09)
ndwi_09_ar = functions.extract_values(shp = file_train09_shp, raster = ndwi_09)
mndwi1_09_ar = functions.extract_values(shp = file_train09_shp, raster = mndwi1_09)
mndwi2_09_ar = functions.extract_values(shp = file_train09_shp, raster = mndwi2_09)
ndbi_09_ar = functions.extract_values(shp = file_train09_shp, raster = ndbi_09)
mndbi_09_ar = functions.extract_values(shp = file_train09_shp, raster = mndbi_09)

train_array_09 = functions.combine_bands_sf(b1 = b1_09_ar, b2 = b2_09_ar, b3 = b3_09_ar, b4 = b4_09_ar, 
	b5 = b5_09_ar, b7 = b7_09_ar, ndvi = ndvi_09_ar, ndwi = ndwi_09_ar, mndwi1 = mndwi1_09_ar, 
	mndwi2 = mndwi2_09_ar, ndbi = ndbi_09_ar, mndbi = mndbi_09_ar,
	multiband_array_file = 'Samples/SLC off/B_train_array_sf_09.npy')


#2011
b1_11_ar = functions.extract_values(shp = file_train11_shp, raster = b1_11)
b2_11_ar = functions.extract_values(shp = file_train11_shp, raster = b2_11)
b3_11_ar = functions.extract_values(shp = file_train11_shp, raster = b3_11)
b4_11_ar = functions.extract_values(shp = file_train11_shp, raster = b4_11)
b5_11_ar = functions.extract_values(shp = file_train11_shp, raster = b5_11)
b7_11_ar = functions.extract_values(shp = file_train11_shp, raster = b7_11)
ndvi_11_ar = functions.extract_values(shp = file_train11_shp, raster = ndvi_11)
ndwi_11_ar = functions.extract_values(shp = file_train11_shp, raster = ndwi_11)
mndwi1_11_ar = functions.extract_values(shp = file_train11_shp, raster = mndwi1_11)
mndwi2_11_ar = functions.extract_values(shp = file_train11_shp, raster = mndwi2_11)
ndbi_11_ar = functions.extract_values(shp = file_train11_shp, raster = ndbi_11)
mndbi_11_ar = functions.extract_values(shp = file_train11_shp, raster = mndbi_11)

train_array_11 = functions.combine_bands_sf(b1 = b1_11_ar, b2 = b2_11_ar, b3 = b3_11_ar, b4 = b4_11_ar, 
	b5 = b5_11_ar, b7 = b7_11_ar, ndvi = ndvi_11_ar, ndwi = ndwi_11_ar, mndwi1 = mndwi1_11_ar, 
	mndwi2 = mndwi2_11_ar, ndbi = ndbi_11_ar, mndbi = mndbi_11_ar,
	multiband_array_file = 'Samples/SLC off/B_train_array_sf_11.npy')


