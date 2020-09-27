# import packages
import cv2
import os
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import sys
import time 
# from datetime import datetime # for benchmarking purpose

# input directory
train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset
# output directory
output_dir = "results"

# User defined threshold
tau = 0.00000017
prior = 0.5



# Define a vectorized function that generates a RGB image's mean value and covariance.
# input shape: X:(num of pixels, 3)
# Output shape: mean-（3,） covariacne-(3x3)
def cal_mean_cov_vectorized(X):
    N, D = X.shape
    mean = X.mean(axis=0)  # compute mean
    cov = np.matmul((X - mean).T, (X - mean)) / (N - 1)  # compute covariance
    return mean, cov




# extract orange pixels from training images
# return Nx3 array
def extract_orange_pixels():
    # store orange pixels, each row is a pixel
    orange_pixels = np.array([])

    for img_name in os.listdir(train_dir):
        if "mask" in img_name:
            continue
        img = cv2.imread(os.path.join(train_dir, img_name))
        # load mask for it
        img_mask = cv2.imread(os.path.join(train_dir, img_name.split(".")[0]+"_mask.png"))
        # reshape to num of rows = num of pixels, num of column = 3 (BGR)
        X = img.transpose(2,0,1).reshape(3,-1).T 
        # reshape and sum mask to 1d array
        X_mask = img_mask.transpose(2,0,1).reshape(3,-1).T.sum(1) 
        # if empty array
        if orange_pixels.size == 0:
            orange_pixels = X[X_mask>50] # get pixels that are not black in mask
        else:
            orange_pixels = np.append(orange_pixels, X[X_mask>20], axis =0)
    return orange_pixels


# train on orange pixels
# param X is orange pixels, Nx3 array 
def train_on_orange_pixels(X):
    mean, cov = cal_mean_cov_vectorized(X)   
    return mean, cov


# test on test images
# param: mean and cov of all orange pixels
def test(orange_mean, orange_cov):
    for img_name in os.listdir(test_dir):
        img = cv2.imread(os.path.join(test_dir, img_name))
        l, w, h = img.shape # original shape of 2D image
        X = img.transpose(2,0,1).reshape(3,-1).T # reshape to num of rows = num of pixels, num of column = 3 (RGB)
        N, D = X.shape
        #img = X.reshape(l, w, -1) # reshape back to 2d image
    
        # calculate likelihood using gaussian distribution
        # each pixel is row of X
        constant_in_likelihood = 1/(math.sqrt(((2*math.pi)**3)* np.linalg.det(orange_cov))) 
        sigma_inv = np.linalg.inv(orange_cov)
        X2 = X-orange_mean
        exponent = (-0.5)*(np.dot(X2, sigma_inv) * X2).sum(1) 
        likelihood = constant_in_likelihood * np.exp(exponent)

        # posterior
        prior = 0.5
        posterior = prior* likelihood 
        
        # mask (reshape back to 2D image)
        mask = posterior.reshape(l, w) 
        #print(mask)
        
        # apply mask
        img[mask < tau] = 0

        ##  show masked img
        image_name = os.path.join(output_dir,"single_gaussian_"+ str(img_name))
        cv2.imshow(image_name, img)
        cv2.imwrite(image_name, img)
        # cv2.waitKey(0)
        print("Finish Generating mask for image ", str(img_name))
    


# if __name__ == "__main__":
#     # load data
#     input_dir = "train_images"  # path to the train image dataset
#     # output directory
#     if not (os.path.isdir("single_gaussian_result")):
#         os.mkdir("single_gaussian_result")
#     output_dir = "single_gaussian_result"
#     # User defined threshold
#     tau = 0.00000000000000001
#     # Number of process
#     print("Starting #", mp.cpu_count(), "of process")
#     pool = mp.Pool()
#     for img in os.listdir(input_dir):
#         pool.apply_async(func=single_gussian, args=(img, input_dir, output_dir, tau,))
#         break
#     pool.close()
#     pool.join()
#     print("All Process are finished")
#     print("Single gussian result are stored in the directory called single_guassian_result")
if __name__ == "__main__":
    # output directory
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    orange_mean, orange_cov = train_on_orange_pixels(extract_orange_pixels())
    print("orange_mean: "+ str(orange_mean)) # BGR, not RGB
    print("orange_cov: \n" + str(orange_cov))
    test(orange_mean, orange_cov)
    print("All images have been processed. Press any key on images to exit.")
    cv2.waitKey(0)
    
