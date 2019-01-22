#run case C, SB+SF, 2011, with 300 trees
import numpy as np 
import functions

"""
locate the Numpy array inputs
"""

#training datasets
#class
train11 = np.load('Samples/SLC off/C_train11.npy')
#surface reflectance value
train_array_11 = np.load('Samples/SLC off/C_train_array_sf_11.npy')

"""
locate the satellite images
"""
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
#read image bands and combine them into img_array

#2011
img_b1_11 = functions.create_band(band = b1_11)
img_b2_11 = functions.create_band(band = b2_11)
img_b3_11 = functions.create_band(band = b3_11)
img_b4_11 = functions.create_band(band = b4_11)
img_b5_11 = functions.create_band(band = b5_11)
img_b7_11 = functions.create_band(band = b7_11)
img_ndvi_11 = functions.create_band(band = ndvi_11)
img_ndwi_11 = functions.create_band(band = ndwi_11)
img_mndwi1_11 = functions.create_band(band = mndwi1_11)
img_mndwi2_11 = functions.create_band(band = mndwi2_11)
img_ndbi_11 = functions.create_band(band = ndbi_11)
img_mndbi_11 = functions.create_band(band = mndbi_11)

img_11 = functions.combine_bands_sf(b1 = img_b1_11, b2 = img_b2_11, b3 = img_b3_11, b4 = img_b4_11, 
	b5 = img_b5_11, b7 = img_b7_11, ndvi = img_ndvi_11, ndwi = img_ndwi_11, mndwi1 = img_mndwi1_11,
	mndwi2 = img_mndwi2_11, ndbi = img_ndbi_11, mndbi = img_mndbi_11,
	multiband_array_file = 'L2 imagery/img_11_sf.npy')

#-----------------------------------------------------------------------------------------------
#train and predict with RF
#400 trees
result_11_300 = functions.train_rf(trees = 300, maxfeatures = None, 
	train_array = train_array_11, gt_array = train11, 
	model_sav = 'Models/C_rf_sf_11_300trees.sav', 
	img = img_11,
	result_array_file = 'L2 imagery/Results/C_result_sf_11_300.npy')
# None == n_features

#-----------------------------------------------------------------------------------------------
# make rasters from results
functions.rasterize(result_array = result_11_300, 
	result_raster = 'L2 imagery/Results/raster_C_result_sf_11_300.tif')
