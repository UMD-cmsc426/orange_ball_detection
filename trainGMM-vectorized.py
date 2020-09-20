import os
import cv2
import numpy as np
import random
import sys
import testGMM
import re
import copy
from gaussian import *
import timeit

# Randomly initialize gaussian distribution
# returns a 1 x 3 matrix, where the first entry is an int, second entry is  a 3x1 matrix, third entry is a 3x3 matrix
def initialize():
    mean = np.asmatrix([[random.randint(1, 255)], [random.randint(1, 255)], [random.randint(1, 255)]])
    # generate a random positive-semidefinete matrix as covariance matrix
    A = np.random.random((3, 3)) * 60
    cov = np.dot(A, A.transpose())
    scaling = (random.random() * 5.0)
    return [scaling, mean, cov]


# return true if MLE converges. Return false otherwise
def check_convergence(total_mean, prev_total_mean, tau):
    sum = np.sum(np.apply_along_axis(np.linalg.norm,1, total_mean - prev_total_mean))
    print("Current Mean: \n", total_mean)
    print("Previous Mean: \n", prev_total_mean)
    print("Current Convergence difference: ", sum)
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
'''
parameters:
k: int, number of guassian distribution
max_iter: int, maximum number of step in optimization
img: np array of image
img_name: strings, the relative path to single image, i.e. "train_images/032.jpg"
'''
def trainGMM(K, max_iter, img, img_name):
    # user defined converge threshold
    tau = 1e-10
    # Structure of para:
    # [[scale,mean,covariance],[scale,mean,covariance],[scale,mean,covariance]...]
    # scale is a int. Mean is a 3x1 matrix. Covariance is a 3x3 matrix
    params = [initialize() for cluster in range(K)]

    # total_mean is the sum of all mean from different clusters
    total_mean = np.full((K, 3, 1), -9999)
    prev_total_mean = np.full((K, 3, 1), 9999)
    iter = 0

    img_w, img_h, img_channel = img.shape

    while iter < max_iter :
        print("--- Starting Iter #",  iter, " ---")
        print("iter: ", iter)
        if check_convergence(total_mean, prev_total_mean, tau):
            break
        start_time = timeit.default_timer()
        # update prev total mean
        prev_total_mean = copy.deepcopy(total_mean)

        # Expectation step - assign points to clusters, get cluster weight
        weights = np.asarray([np.zeros((img.shape[0], img.shape[1])) for _ in range(K)])
        for cluster in range(K):
            # cumulated weights add up all weights on a given pixel -- serving as denominator
            cumulated_weights = np.zeros((img_w, img_h))
            cluster_scaling, cluster_mean, cluster_cov = params[cluster]

            # TODO: PROBLEM: the likelihood are the same across the image in one cluster
            # Calculate likelihood
            constant_in_likelihood = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cluster_cov)))
            sigma_inv = np.linalg.inv(cluster_cov)
            # populate mean here
            flatted_mean = np.repeat(cluster_mean, img_w*img_h, axis=0)
            flatted_mean = np.asarray(flatted_mean)
            populated_mean= flatted_mean.reshape((img_channel,img_w,img_h))
            populated_mean = np.moveaxis(populated_mean, 0, -1)
            mean_diff = img - populated_mean

            # apply limit to exponent to prevent any overflow or underflow
            exponent = np.minimum(np.maximum(along_axis(mean_diff, sigma_inv), 1e-60), 1e20)
            likelihood =constant_in_likelihood * np.exp(exponent)

            # weight for a single cluster
            cluster_weights = cluster_scaling * likelihood
            cumulated_weights += cluster_weights
            weights[cluster] = cluster_weights
            # Sanity Check
            if (weights[cluster] == 0).all():
                raise Exception("all weights are zero")
            if (weights[cluster] == 0).any():
                raise Exception("one of the weights is zero")
            # ---------end of vectorize ---------

        # broadcast cumulated_weights to shape [K, img_w, img_h]
        weights = weights/(np.tile(cumulated_weights[np.newaxis,:], (K,1,1)))
        # Maximization step - get new scaling, mean, and cov for each cluster
        for cluster in range(K):
            mean_sum = np.zeros((img_w, img_h)) # sums all weight*pixel RGB value on image
            cov_sum = np.zeros((3, 3))
            sum_weights = np.sum(weights[cluster]) # sum of all the weights given a cluster

            # Calculate new mean: boardcast weights[cluster] to three channels, then perform element-wise mutiplication with img, which is
            # is 3d np array. Then calculate the sum of all elements three channels
            broad_casted_weight =  (np.tile(weights[cluster][:,:, np.newaxis],(1, 1, img_channel))) # shape: (img_h, img_w, 3)
            weighted_pixel = np.multiply(img, broad_casted_weight) # shape: (img_h, img_w, 3)
            new_mean = np.asarray([[np.sum(weighted_pixel[:,:,0])],[np.sum(weighted_pixel[:,:,1])],[np.sum(weighted_pixel[:,:,2])]]) / sum_weights
            # TODO: Calculate new covariance using vectorization:
            # populate new mean to shape (img_h, img_w, 3)
            # flatted_mean = np.repeat(cluster_mean, img_w * img_h, axis=0)
            # flatted_mean = np.asarray(flatted_mean)
            # populated_mean = flatted_mean.reshape((img_w, img_h, img_channel))
            # _mean_diff = img - populated_mean
            # new_covariance = np.apply_along_axis(covariance_vectorized, 2, _mean_diff) # new_covariance shape: (img_g, img_w, 1, 3,3)
            # raise Exception("Stop")

            # ------- need vectorize --------
            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    # calculate covariance
                    cov_sum += np.multiply(weights[cluster][w][h], (pix - new_mean))@((pix - new_mean).T)
            # ---------end of vectorize ---------
            new_cov = np.divide(np.double(cov_sum), np.double(sum_weights))
            new_scaling = sum_weights / (img_w*img_h)
            mean_sum += mean_sum

            total_mean[cluster] = new_mean
            # update model
            params[cluster] = (new_scaling, new_mean, new_cov)

        end_timer = timeit.default_timer()
        print("Iter #", iter, "use time: ", end_timer-start_time, "s")
        print("---  End of Iter  ---\n")
        iter += 1
    # store weights to .npy
    if not os.path.exists("weights"):
        os.mkdir("weights")
    else:
        digit = re.findall(r'\d+\d+\d*',img_name)
        file_name = str(digit[0])+"_weight.npy"
        with open(os.path.join("weights",file_name), "wb") as f:
            np.save(f,params)


if __name__ == "__main__":
    #np.seterr(all='raise')
    input_dir = "train_images"
    for img_name in os.listdir(input_dir):
        img = os.path.join(input_dir, img_name)
        trainGMM(5,100,cv2.imread(img), img)
        print("Finish Training for ", img)
        break


