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
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for filePath in sorted(os.listdir(path)):
        # 将文件名和文件扩展名分离，[0]是文件名，[1]是文件扩展名
        # fileExt = os.path.splitext(filePath)[1]
        # if fileExt in [".jpg", ".jpeg"]:

            # Add to array of FaceImages
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)
        # imread 的output is (192,163,3)的np.array, dtype = uint8

            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:
                # Convert image to floating point
                # I have no idea about this operation, but all the sample code on the Internet have the operation
                im = np.float32(im) / 255.0
                # Add image to list
                images_set.append(im)
                # # Flip image
                # imFlip = cv2.flip(im, 1);
                # # Append flipped image
                # FaceImages.append(imFlip)
    sum_faceImages = int(len(images_set))
    # Exit if no image found
    if sum_faceImages == 0:
        print("No FaceImages found")
        sys.exit(0)

    print(str(sum_faceImages) + " files read.")
    return images_set


# Create original matrix from a list of FaceImages
# There is no need to build Covariance matrix,
# because the function 'cv2.PCACompute' will do it and return mean & eigenvector
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
    matrix_face = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    for i in range(0, numImages):
        # flatten() is numpy.ndarray.flatten, which convert arrays into one-dimension array
        # 默认是按row将多维的array进行拼接成one-dimension array
        # The function convert each face image into one-dimension row vector (array)
        image = images[i].flatten()
        # fill these one-dimension row vector into the matrix created before.
        matrix_face[i, :] = image

    print("DONE")
    return matrix_face

# Add the weighted eigenFaces to the mean (row vector)
def createNewFace(mean,dirname,num_eigenfaces,eigenVectors):
    '''

    :param mean: (1,93798) the AverageVector got from cv2.PCACompute
    :param dirname: the directory of new faceimage
    :param num_eigenfaces: the number of eigenvector
    :param eigenVector: the list of eigenvectors
    :return: new faceimage vector with weight
    '''
    print("Loading FaceImages from " + dirname, end="...")
    # Start with the mean image(come from cv2.PCACompute)
    output = mean

    new_faceimage = cv2.imread(dirname)
    # print('size of new_faceimage is', np.shape(new_faceimage))
    new_faceimage = np.float32(new_faceimage) / 255.0
    new_faceimage = new_faceimage.flatten()
    # print('size of new_faceimage is', np.shape(new_faceimage))

    for i in range(0, num_eigenfaces):
        weight = (new_faceimage-mean).dot(eigenVectors[i].T)
        # print('weight is ',weight)

        # new faceimages (row vector) with weighted
        output = np.add(output, eigenVectors[i] * weight)

    print("DONE")
    return output

def CalculateMSE(mean,dirname,num_eigenfaces,eigenVectors):

    print("Calculating MSE from" + dirname, end="...")
    new_faceimage = cv2.imread(dirname)
    new_faceimage = np.float32(new_faceimage) / 255.0
    new_faceimage = new_faceimage.flatten()
    new_faceimage_reshape = new_faceimage.reshape((1, 93798))

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
    plt.plot(mean_SE)
    plt.show()


if __name__ == '__main__':

    # Number of EigenFaces
    num_eigenfaces = 190

    # # Maximum weight
    # MAX_SLIDER_VALUE = 255

    # Directory containing FaceImages
    dirName = "Netural_Images"

    # Read FaceImages
    images = readImages(dirName)

    # Size of FaceImages, 为了让后续生成的一系列row vector通过reshape变回图像的（193,162,3）的np.array
    sz = images[0].shape
    # print('shape of sz is', sz)



    # Create original faceimages matrix for PCA, each faceimage is a row vector
    matrix_faces = createDataMatrix(images)
    matrix_faces_T = matrix_faces.T
    print('Shape of matrix_faces is ', np.shape(matrix_faces))
    # covar, mean_1 = cv2.calcCovarMatrix(matrix_faces, mean=None, flags=cv2.COVAR_ROWS|cv2.COVAR_SCRAMBLED)
    covar, mean = cv2.calcCovarMatrix(matrix_faces, mean=None, flags=cv2.COVAR_SCALE | cv2.COVAR_ROWS| cv2.COVAR_SCRAMBLED)
    # mean_x is same as mean_1,  cv2.COVAR_SCALE 相当于1/M, because the default is 1

    print("shape of mean is ", np.shape(mean))
    print("shape of covar is ", np.shape(covar))
    print('covar is', covar)

    eVal, eVec = cv2.eigen(covar, True)[1:]

    # 对eVec进行一系列操作，（ui）T * （A）T，再进行normalization，因为OpenCV都是row vector运算，所以跟论文中相当于都要进行转置
    eVec = cv2.gemm(eVec, matrix_faces - mean_x, 1, None, 0)
    eVec = np.apply_along_axis(lambda n: cv2.normalize(n,n).flat, 1, eVec)
#     print("shape of eVal is ", np.shape(eVal))
#     # print(eVal)
#     # mean_SE = pd.DataFrame.from_dict(mean_SE, orient='index', columns=['values'])
#     # plt.plot(eVal)
#     # plt.show()
#
#     print("shape of eVec is ", np.shape(eVec))
#
#     #     # Compute the eigenvectors from the Matrix (modified from matrix_faces)
# #     # we need to make each data(face images) as row vectors
# #     print("Calculating PCA ", end="...")
#     mean_2, eigenVectors = cv2.PCACompute(matrix_faces, mean=None)
#     print('mean_1 is', mean_1)
#     print('mean_x is',mean_x)
#     print('mean_2 is', mean_2)
#
#     print('shape of eVec is', np.shape(eVec))
#     print('shape of eigenVector is', np.shape(eigenVectors))
#
#     print('eVec is', eVec[0])
#     print('eigenVector is', eigenVectors[0])
#     print(eVec-eigenVectors)







#     print("DONE")
#     print('mean is ',mean)
#     print('shape of mean is',np.shape(mean))
#     print('type of eigenVectors is ', type(eigenVectors))
#     print('size of eigenVectors is', np.shape(eigenVectors))
#     # print('1st eigenvector is', eigenVectors[0])
#     # print('number of eigenvectors is',len(eigenVectors))
#     # print('the norm of 1st eigenvectors is', np.linalg.norm(eigenVectors[2]))
#
#     # Create window for displaying Mean Face
#     cv2.namedWindow("AverageFace", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("AverageFace", mean.reshape(sz))
#
# # load new faceimages, and use eigenvector to build new faceimage with weight
#     new_faceimage = createNewFace(mean, '101b.jpg', num_eigenfaces, eigenVectors)
#     print('type of new_faceimages is ', type(new_faceimage))
#     print(new_faceimage)
# # show the new_faceimage with weight
#     cv2.namedWindow('new_faceimage', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('new_faceimage',new_faceimage.reshape(sz))
#
#     CalculateMSE(mean, '101b.jpg', num_eigenfaces, eigenVectors)


    cv2.waitKey(0)
    cv2.destroyAllWindows()