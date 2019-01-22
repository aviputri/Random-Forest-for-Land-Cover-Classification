#run case C, SB+SF, 2009, with 400 trees
import numpy as np 
import functions

"""
locate the Numpy array inputs
"""

#training datasets
#class
train09 = np.load('Samples/SLC off/C_train09.npy')
#surface reflectance value
train_array_09 = np.load('Samples/SLC off/C_train_array_sf_09.npy')

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
img_ndwi_09 = functions.create_band(band = ndwi_09)
img_mndwi1_09 = functions.create_band(band = mndwi1_09)
img_mndwi2_09 = functions.create_band(band = mndwi2_09)
img_ndbi_09 = functions.create_band(band = ndbi_09)
img_mndbi_09 = functions.create_band(band = mndbi_09)

img_09 = functions.combine_bands_sf(b1 = img_b1_09, b2 = img_b2_09, b3 = img_b3_09, b4 = img_b4_09, 
	b5 = img_b5_09, b7 = img_b7_09, ndvi = img_ndvi_09, ndwi = img_ndwi_09, mndwi1 = img_mndwi1_09,
	mndwi2 = img_mndwi2_09, ndbi = img_ndbi_09, mndbi = img_mndbi_09,
	multiband_array_file = 'L2 imagery/img_09_sf.npy')

#-----------------------------------------------------------------------------------------------
#train and predict with RF
#400 trees
result_09_400 = functions.train_rf(trees = 400, maxfeatures = None, 
	train_array = train_array_09, gt_array = train09, 
	model_sav = 'Models/C_rf_sf_09_400trees.sav', 
	img = img_09,
	result_array_file = 'L2 imagery/Results/C_result_sf_09_400.npy')
# None == n_features

#-----------------------------------------------------------------------------------------------
# make rasters from results
functions.rasterize(result_array = result_09_400, 
	result_raster = 'L2 imagery/Results/raster_C_result_sf_09_400.tif')
