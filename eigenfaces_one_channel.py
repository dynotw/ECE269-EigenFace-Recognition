# The following codes are modified from learn opencv
# The code uses the face images as row vector rather than column vector
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd

# Read FaceImages from the directory
def readImages(path):
    print("Reading FaceImages from " + path, end="...")
    # Create array of array of FaceImages.
    images_set = []
    # List all files in the directory and read points from text files one by one
    # os.listdir(), return a list including all files and documents in the specific files
    for filePath in sorted(os.listdir(path)):

            # Add to array of FaceImages
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath,flags=cv2.IMREAD_GRAYSCALE)
        # imread 的output is (192,163,3)的np.array, dtype = uint8

            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:
                # Convert image to floating point
                # I have no idea about this operation, but all the sample code on the Internet have the operation
                im = np.float32(im)
                # Add image to list
                images_set.append(im)
    sum_faceImages = int(len(images_set))
    print(str(sum_faceImages) + " files read.")
    return images_set


# Create original matrix from a list of FaceImages
# cv2.PCACompute is design defaultly for row vector
def createDataMatrix(images):
    # The Input is not truely original face FaceImages
    # these face FaceImages are dealt, when we read them (function: 'readImages')
    print("Creating data matrix", end=" ... ")
    ''' 
    Allocate space for all FaceImages in one data matrix.
    The size of the data matrix is ( w  * h  * 3, numImages )
    w = width of an image in the dataset.
    h = height of an image in the dataset.
    3 is for the 3 color channels.
    '''
    # In the project, all face FaceImages are grey FaceImages with only 1 color channel
    # sz[2] always be 1

    # len() is calculating the number of face FaceImages in the training set
    numImages = len(images)
    # Create a matrix, according to 1st face image,
    # because shapesize of all face FaceImages in the training set are same, so only use 1st face
    sz = images[0].shape

    # create a matrix to describe all face FaceImages in the training set
    # numImages is the number of rows (vector), each face FaceImages will be described by a row vector
    # sz[0]*sz[1]*sz[2] is the dimension of the row vector
    matrix_face = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, numImages):
        # flatten() is numpy.ndarray.flatten, which convert arrays into one-dimension array
        # use row to make up one-dimension numpy array
        # The function convert each face image into one-dimension row vector (array)
        image = images[i].flatten()
        # fill these one-dimension row vector into the matrix created before.
        matrix_face[i, :] = image

    print("DONE")
    return matrix_face

# Add the weighted eigenFaces to the mean (row vector)
def createNewFace(mean,dirname,num_eigenfaces,eigenVectors):

    print("Loading FaceImages from " + dirname, end="...")
    # Start with the mean image(come from cv2.PCACompute)
    output = mean

    new_faceimage = cv2.imread(dirname,flags=cv2.IMREAD_GRAYSCALE)

# rotated image
    height, width = new_faceimage.shape[:2]
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, 225, 1)
    new_faceimage = cv2.warpAffine(new_faceimage, rot_mat, (new_faceimage.shape[1], new_faceimage.shape[0]))

    # # resieze shape of input image
    # new_faceimage = cv2.resize(new_faceimage,(193,162))

    new_faceimage = np.float32(new_faceimage)
    new_faceimage = new_faceimage.flatten()
    # print('size of new_faceimage is', np.shape(new_faceimage))

    for i in range(0, num_eigenfaces):
        weight = (new_faceimage-mean).dot(eigenVectors[i].T)

        # new faceimages (row vector) with weighted
        output = np.add(output, eigenVectors[i] * weight)

    print("DONE")
    return output

def CalculateMSE(mean,dirname,num_eigenfaces,eigenVectors):

    print("Calculating MSE from" + dirname, end="...")
    new_faceimage = cv2.imread(dirname,flags=cv2.IMREAD_GRAYSCALE)

    # # for resize no human face image
    # new_faceimage = cv2.resize(new_faceimage,(193,162))

# rotate image
    height, width = new_faceimage.shape[:2]
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, 225, 1)
    new_faceimage = cv2.warpAffine(new_faceimage, rot_mat, (new_faceimage.shape[1], new_faceimage.shape[0]))

    new_faceimage = np.float32(new_faceimage)
    new_faceimage = new_faceimage.flatten()
    new_faceimage_reshape = new_faceimage.reshape((1, 31266))

    mean_SE = dict()
    for i in range(1,num_eigenfaces+1):

        output = mean

        for j in range(0, i):
            weight = (new_faceimage - mean).dot(eigenVectors[j].T)

            output = np.add(output, eigenVectors[j] * weight)

        mean_SE1=mse(new_faceimage_reshape,output)
        # print(MSE1)
        mean_SE[i-1]=mean_SE1

    print("DONE")
    print('The number of MSE is', len(mean_SE))

    mean_SE = pd.DataFrame.from_dict(mean_SE, orient='index', columns=['values'])
    print(mean_SE)
    print(type(mean_SE))
    plt.plot(mean_SE)
    # ax = plt.gca()
    # x_major_locator=plt.MultipleLocator(10)
    # ax.xaxis.set_major_locator(x_major_locator)

    plt.text(190, mean_SE['values'][189], (190, round(mean_SE['values'][189],2)))

    plt.show()


if __name__ == '__main__':

# Number of EigenFaces
    num_eigenfaces = 190

# Directory containing FaceImages
    dirName = "Netural_Images"

# Read FaceImages
    images = readImages(dirName)

# Size of FaceImages, in order to resize all the following row vector into (1,32166)
    sz = images[0].shape

# Create original faceimages matrix for PCA, each faceimage is a row vector
    matrix_faces = createDataMatrix(images)
    print('Shape of matrix_faces is ', np.shape(matrix_faces))
    matrix_faces_T = matrix_faces.T

    # get the covariance matirx and mean
    covar, mean = cv2.calcCovarMatrix(matrix_faces, mean=None, flags=cv2.COVAR_SCALE | cv2.COVAR_ROWS| cv2.COVAR_SCRAMBLED)
    # mean_x is same as mean_1,  cv2.COVAR_SCALE is equal to 1/M, because the default is 1

    print("shape of mean is ", np.shape(mean))
    print("shape of covar is ", np.shape(covar))

# get eigenvalues and eigenvectors
    print("Calculating PCA ", end="...")
    eVal, eVec = cv2.eigen(covar, True)[1:]
    svd = np.sqrt(190*eVal)
    print(svd)

#  to make some operations on eVec，1. （ui）T * （A）T，2. normalize，
#  OpenCV uses row operations rather than column operations, so we must transpose the matrixs
    eVec = cv2.gemm(eVec, matrix_faces - mean, 1, None, 0)
    eVec = np.apply_along_axis(lambda n: cv2.normalize(n,n).flat, 1, eVec)
    print('DONE')

    plt.plot(svd)

    ax = plt.gca()
    x_major_locator=plt.MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    min_indx=np.argmin(svd)#min value index
    show_min='['+str((min_indx)+1)+','+str(svd[min_indx])+']'
    plt.annotate(show_min,xytext=(min_indx,svd[min_indx]),xy=(min_indx,svd[min_indx]))
    plt.show()

# # Create window for displaying Mean Face
#     cv2.namedWindow("AverageFace", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("AverageFace", mean.reshape(sz))

# load new faceimages, and use eigenvector to build new faceimage with weight
    new_faceimage = createNewFace(mean, '1a.jpg', num_eigenfaces, eVec)
    print('shape of new_faceimage',np.shape(new_faceimage))

# # show the new_faceimage with weight
#     cv2.namedWindow('new_faceimage', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('new_faceimage',new_faceimage.reshape(sz))
#     # cv2.imwrite(r'C:\Users\83414\Desktop\ECE 269\Project\225_f.jpg',new_faceimage.reshape(sz)*255)

    CalculateMSE(mean, '1a.jpg', num_eigenfaces, eVec)


    cv2.waitKey(0)
    cv2.destroyAllWindows()