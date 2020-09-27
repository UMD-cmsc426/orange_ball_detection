import os
import cv2
import numpy as np
import math
from gaussian import *

train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset
output_dir = os.path.join("results","GMM")
# test on test images
# params: [[scale,mean,covariance],[scale,mean,covariance], [scale,mean,covariance]...]
# scale is a int. Mean is a 3x1 ndarray. Covariance is a 3x3 ndarray
def testGMM(params, tau_test, K, prior):
    for img_name in os.listdir(test_dir):
        img = cv2.imread(os.path.join(test_dir, img_name))
        l, w, h = img.shape # original shape of 2D image
        X = img.transpose(2,0,1).reshape(3,-1).T # reshape to num of rows = num of pixels, num of column = 3 (RGB)
        N, D = X.shape
        #img = X.reshape(l, w, -1) # reshape back to 2d image
        likelihood = np.zeros(N)
        
        for cluster in range(K):
            cluster_scaling, cluster_mean, cluster_cov = params[cluster]
            # calculate likelihood using gaussian distribution
            # each pixel is row of X
            constant_in_likelihood = 1/(math.sqrt(((2*math.pi)**3)* np.linalg.det(cluster_cov))) 
            sigma_inv = np.linalg.inv(cluster_cov)
            X2 = X-cluster_mean
            exponent = (-0.5)*(np.dot(X2, sigma_inv) * X2).sum(1) 
            cluster_likelihood = cluster_scaling *constant_in_likelihood * np.exp(exponent)
            likelihood += cluster_likelihood
        
        # posterior
        posterior = prior* likelihood 

        # mask (reshape back to 2D image)
        mask = posterior.reshape(l, w) 
        #print(mask)
        
        # apply mask
        img[mask < tau_test] = 0

        ##  show masked img
        image_name = os.path.join(output_dir,"GMM_"+ str(img_name))
        # cv2.imshow(image_name, img)
        cv2.imwrite(image_name, img)
        print("Finish Generating mask for image ", str(img_name))

    print("All Images Completed")
    print("Test results are stored at location /result/GMM")
    return None, None, None


if __name__ == "__main__":
    print("Please run GMM.py.")
