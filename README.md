# Create_LMDB_from_Numpy

This file creates an LMDB file from numpy files. The application is face varification in which the
LMDB file will be created from pairs of images. The genuine pairs get label "1" and the imposter
pairs get label "0". The genuine files contains "gen" in their file name.
For this process, the numpy files are HOG features
but they basically acn be anything saved in numpy files.
This file do the following:
1 - load all the ".npy" data from a folder and create a big numpy array
2 - Create an LMDB file from that numpy array
