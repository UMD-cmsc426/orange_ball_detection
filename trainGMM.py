import os
import cv2
import numpy as np
import random
from gaussian import *
import math

train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset


# Randomly initialize gaussian distribution
# returns a 1 x 3 matrix, 
# where the first entry is an int, second entry is  a 1x3 matrix, 
# third entry is a 3x3 matrix
def initialize():
    mean = np.array([random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)])
    # generate a random positive-semidefinete matrix as covariance matrix
    A = np.random.random((3, 3)) * 60
    cov = np.dot(A, A.transpose())
    scaling = (random.random() * 5.0)
    return [scaling, mean, cov]


# Select a random pixel for initialization
# returns a 1 x 3 matrix, 
# where the first entry is an int, second entry is  a 1x3 matrix, 
# third entry is a 3x3 matrix
def initialize_use_pixel(X):
    N, D = X.shape
    mean = X[random.randint(0, N-1)]
    # generate a random positive-semidefinete matrix as covariance matrix
    A = np.random.random((3, 3)) * 60
    cov = np.dot(A, A.transpose())
    scaling = (random.random() * 5.0)
    return [scaling, mean, cov]




# return true if MLE converges. Return false otherwise
def check_convergence(total_mean, prev_total_mean, tau, iter):
    sum = np.sum(np.apply_along_axis(np.linalg.norm,1, total_mean - prev_total_mean))
    # print("Current Mean: \n", total_mean)
    # print("Previous Mean: \n", prev_total_mean)
    #print("Iter " + str(iter) + ": Convergence difference= ", sum)
    return sum <= tau

# In order to apply "np.apply_along_aixs" function with argument input, we have to define our own along_axis function
def along_axis(M, argument):
    return np.apply_along_axis(expoent_vectorized, 2, M, argument)

# mean_diff here is transposed, i.e. mean_diff_transposed.shape = (1, 3)
def expoent_vectorized(mean_diff_transposed, sigma_inv):
    _mean_diff = np.asmatrix(mean_diff_transposed).T
    result = np.asscalar((-0.5) * ((_mean_diff.T) @ sigma_inv @ _mean_diff))
    return result

def covariance_vectorized (mean_diff_transposed):
    mean_diff = np.asmatrix(mean_diff_transposed).T
    return [mean_diff@(mean_diff.T)]



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



'''
parameters:
K: int, number of guassian distribution
max_iter: int, maximum number of step in optimization
X: Nx3 array of all orange pixels
'''
def trainGMM(K, max_iter, X, tau_train):
    # Structure of params:
    # [[scale,mean,covariance],[scale,mean,covariance], [scale,mean,covariance]...]
    # scale is a int. Mean is a 3x1 ndarray. Covariance is a 3x3 ndarray
    params = [initialize_use_pixel(X) for cluster in range(K)]
    prev_mean = None
    iter = 0
    N, D = X.shape
    
    while iter < max_iter :
        weights = np.array([]) # i-th row is weight for i-th cluster
        # print("--- Starting Iter #",  iter, " ---")
        if iter>0 and check_convergence(np.array([cluster_mean for cluster_scaling, cluster_mean, cluster_cov in params]), prev_mean, tau_train, iter): break
        
        # Expectation step - get cluster weight
        for cluster in range(K):
            cluster_scaling, cluster_mean, cluster_cov = params[cluster]
            # if there is error, then re-initialize
            if np.isnan(cluster_mean).any():
                return trainGMM(K, max_iter, X, tau_train)
            # Calculate likelihood, each pixel is row of X
            try:
                constant_in_likelihood = 1/(math.sqrt(((2*math.pi)**3)* np.linalg.det(cluster_cov))) 
                sigma_inv = np.linalg.inv(cluster_cov)
            except:
                # catch error
                print("division by zero or singular value")
                print(cluster_mean)
                print(cluster_cov)
            X_diff = X-cluster_mean
            exponent = (-0.5)*(np.dot(X_diff, sigma_inv) * X_diff).sum(1) 
            likelihood = constant_in_likelihood * np.exp(exponent)
            # weight for a single cluster
            cluster_weights = cluster_scaling * likelihood
            # append cluster weight to weights
            weights = cluster_weights.reshape(1,-1) if weights.size == 0 else np.append(weights, cluster_weights.reshape(1,-1), axis=0)
        weights = weights/weights.sum(0)
        
        # update prev total mean
        prev_mean = np.array([cluster_mean for cluster_scaling, cluster_mean, cluster_cov in params])
        
        # Maximization step - get new scaling, mean, and cov for each cluster
        for cluster in range(K):
            # sum of all the weights given a cluster
            sum_weights = np.sum(weights[cluster]) 
            # Calculate new mean: 
            new_mean = np.multiply(X, weights[cluster].reshape(N,1)).sum(0) / sum_weights
            # if a cluster's R value is too close to 255, change it to 254.5. A R value of 255 will cause determinant to be 0
            if new_mean[2]>254.5: new_mean[2] = 254.5
            # calculate new cov
            X_diff = X-params[cluster][1]
            new_cov = np.dot(np.multiply(X_diff, weights[cluster].reshape(N,1)).T, X_diff)/ sum_weights
            # calculate new scale
            new_scaling = sum_weights/N
            # update params
            params[cluster] = (new_scaling, new_mean, new_cov)

        iter += 1
        
    # print("final means")
    # print(np.array([cluster_mean for cluster_scaling, cluster_mean, cluster_cov in params]))
    return params


if __name__ == "__main__":
    # User defined threshold
    tau_train = 0.7
    tau_test = 0.0000004
    prior = 0.5
    K = 20
    max_iter = 500
    train_dir = "train_images"# path to the train image dataset

    # load training data
    X = extract_orange_pixels()
    # train
    params = trainGMM(K, max_iter, X, tau_train)
    print("Finish Training")


