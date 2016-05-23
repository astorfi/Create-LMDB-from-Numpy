
import os
import glob
import sys

import multiprocessing
import random
import dircache
import lmdb
import numpy as np
import caffe
from caffe.proto import caffe_pb2

"""
This file creates an LMDB file from numpy files. The application is face varification in which the
LMDB file will be created from pairs of images. The genuine pairs get label "1" and the imposter
pairs get label "0". The genuine files contains "gen" in their file name.

For this process, the numpy files are HOG features
but they basically acn be anything saved in numpy files.

This file do the following:
1 - load all the ".npy" data from a folder and create a big numpy array
2 - Create an LMDB file from that numpy array
"""



# Source and destination paths(both with absolute path).
this_path = os.path.dirname(os.path.abspath(__file__))
src_folder_path = 'absolute/path/to/folder/of/files'
dst_folder_path = 'absolute/path/to/folder/that/LMDB/file/will/be/created/LMDB_FILE_NAME'

# # Getting the number of cores for parallel processing
# num_cores = multiprocessing.cpu_count()
# print('Total number of cores', num_cores)

# Getting the number of files in the folder
num_files = (len([name for name in os.listdir(src_folder_path) if os.path.isfile(os.path.join(src_folder_path, name))]))
print("Number of files = ",num_files)

RandFile = random.choice(dircache.listdir(src_folder_path))
FileShape = np.load(os.path.join(src_folder_path,RandFile)).shape
print("File shape: ",FileShape)


# Predifining
num_hog_features = FileShape[1] # Number of HOG features
N = num_files
X = np.zeros((N, 1, num_hog_features, 2), dtype=np.float32) # the number 2 is because the hog features of a pairs should be considered separately
y = np.zeros(N, dtype=np.int64)#
FileNum = np.zeros(N, dtype=np.int64)

# Initialize a counter.
counter = 0

# We should shuffle the order to save the files in LMDB format.
Rand_idx = np.random.permutation(range(X.shape[0]))

# Reading all the numpy files
for f in glob.glob(os.path.join(src_folder_path, "*.npy")):
    # Load numpy file
    numpy_array = np.load(f)

    # # Uncomment if the files have naming order and you want to save the naming order too.
    # file_num = os.path.basename(os.path.basename(f).split('_')[1]).split('.')[0]    # This gets the number of file
    # FileNum[Rand_idx[counter]] = file_num

    # Save to new big vector X.
    X[Rand_idx[counter],:,:,:] = numpy_array

    # If the pairs is genuine, then it has a "gen" in its name.
    if 'gen' in f:
        y[Rand_idx[counter]] = 1
    if counter % 100 == 0:
        # print("Processing file: {}".format(f))
        print("Processing %d pairs" % counter)
    counter = counter + 1

# Create a big array in order to turn into LMDB
X = np.delete(X, np.s_[counter:X.shape[0]],0)  # delete extra pre-allocated space

# LMDB GENERATION
ExtraMem = 100000   # Rule of thumb: the bigger the better if not out of memory!!
env = lmdb.open(dst_folder_path, map_size=ExtraMem * counter)

lmdb_file = dst_folder_path
batch_size = 256

lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
lmdb_file = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

batch_num = 1
file_to_LMDB_num = 0
for i in range(N):

    #prepare the data and label
    data = X[i,:,:,:].astype(float)
    label = int(y[i])

    # save in datum
    datum = caffe.io.array_to_datum(data, label)
    key = '{:0>8d}'.format(file_to_LMDB_num)
    lmdb_file.put(key, datum.SerializeToString() )

    # write batch(batch size is flexible)
    if file_to_LMDB_num % batch_size == 0 and file_to_LMDB_num > 0:
        lmdb_file.commit()
        lmdb_file = lmdb_env.begin(write=True)
        print ("Generating batch {}...".format(batch_num))
        batch_num += 1

    # Increasing the counter
    file_to_LMDB_num += 1

# write last batch(because the number of files cannot be necessary divisive by the batch size)
if (file_to_LMDB_num) % batch_size != 0:
    lmdb_file.commit()
    print('generating last batch ...')
    print (file_to_LMDB_num)
