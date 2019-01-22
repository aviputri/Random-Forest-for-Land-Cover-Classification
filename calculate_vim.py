import functions
import numpy as np

"""
locate training datasets
"""

#case A
roi09_a = np.load('Samples/SLC off/A_train09.npy')
sat09_a = np.load('Samples/SLC off/A_train_array_sf_09.npy')

roi11_a = np.load('Samples/SLC off/A_train11.npy')
sat11_a = np.load('Samples/SLC off/A_train_array_sf_11.npy')

#case B
roi09_b = np.load('Samples/SLC off/B_train09.npy')
sat09_b = np.load('Samples/SLC off/B_train_array_sf_09.npy')

roi11_b = np.load('Samples/SLC off/B_train11.npy')
sat11_b = np.load('Samples/SLC off/B_train_array_sf_11.npy')

#case A
roi09_c = np.load('Samples/SLC off/C_train09.npy')
sat09_c = np.load('Samples/SLC off/C_train_array_sf_09.npy')

roi11_c = np.load('Samples/SLC off/C_train11.npy')
sat11_c = np.load('Samples/SLC off/C_train_array_sf_11.npy')

"""
"""

print("Case A")
print("2009")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_a, gt_array = roi09_a)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat09_a, gt_array = roi09_a)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat09_a, gt_array = roi09_a)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat09_a, gt_array = roi09_a)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_a, gt_array = roi09_a)

"""
"""
print(" ")
print("2011")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_a, gt_array = roi11_a)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat11_a, gt_array = roi11_a)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat11_a, gt_array = roi11_a)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat11_a, gt_array = roi11_a)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_a, gt_array = roi11_a)

"""
"""
print(" ")
print("--------------------------------------")
print("Case B")
print("2009")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_b, gt_array = roi09_b)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat09_b, gt_array = roi09_b)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat09_b, gt_array = roi09_b)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat09_b, gt_array = roi09_b)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_b, gt_array = roi09_b)

"""
"""
print(" ")
print("2011")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_b, gt_array = roi11_b)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat11_b, gt_array = roi11_b)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat11_b, gt_array = roi11_b)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat11_b, gt_array = roi11_b)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_b, gt_array = roi11_b)

"""
"""
print(" ")
print("--------------------------------------")
print("Case C")
print("2009")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_c, gt_array = roi09_c)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat09_c, gt_array = roi09_c)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat09_c, gt_array = roi09_c)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat09_c, gt_array = roi09_c)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat09_c, gt_array = roi09_c)

"""
"""
print(" ")
print("2011")
print("100 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_c, gt_array = roi11_c)

print("200 trees")
functions.vim(trees = 200, maxfeatures = None, train_array = sat11_c, gt_array = roi11_c)

print("300 trees")
functions.vim(trees = 300, maxfeatures = None, train_array = sat11_c, gt_array = roi11_c)

print("400 trees")
functions.vim(trees = 400, maxfeatures = None, train_array = sat11_c, gt_array = roi11_c)

print("500 trees")
functions.vim(trees = 100, maxfeatures = None, train_array = sat11_c, gt_array = roi11_c)

