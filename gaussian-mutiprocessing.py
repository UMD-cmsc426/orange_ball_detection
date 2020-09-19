# import packages
import cv2
import os
import numpy as np
import math
import multiprocessing as mp
import time

# Define a function that generates a RGB image's mean value and covariance.
# Output shape: mean-（3x1） covariacne-(3x3)
def cal_mean_cov(img):
    l, w, h = img.shape
    mean = [[np.sum(img[:, :, 0]) / (l * w)], [np.sum(img[:, :, 1]) / (l * w)], [np.sum(img[:, :, 2]) / (l * w)]]
    cov = np.zeros((3, 3), )
    for width in range(len(img[:, 0, 0])):
        for length in range(len(img[0, :, 0])):
            RGB_value = [[img[width][length][0]], [img[width][length][1]], [img[width][length][2]]]
            cov = cov + (np.asmatrix(RGB_value) - np.asmatrix(mean)) @ (np.asmatrix(RGB_value) - np.asmatrix(mean)).T
    cov = cov / (l * w)
    return mean, cov


# Define a vectorized function that generates a RGB image's mean value and covariance.
# input shape: X:(num of pixels, 3)
# Output shape: mean-（3,） covariacne-(3x3)
def cal_mean_cov_vectorized(X):
    N, D = X.shape
    mean = X.mean(axis=0)  # compute mean
    cov = np.matmul((X - mean).T, (X - mean)) / (N - 1)  # compute covariance
    return mean, cov


def single_gussian(img_name, input_dir, output_dir, tau):
    img = cv2.imread(os.path.join(input_dir, img_name))
    l, w, h = img.shape  # original shape of 2D image
    X = img.transpose(2, 0, 1).reshape(3, -1).T  # reshape to num of rows = num of pixels, num of column = 3 (RGB)
    N, D = X.shape
    # img = X.reshape(l, w, -1) # reshape back to 2d image

    # vectorized algorithm
    mean, cov = cal_mean_cov_vectorized(X)

    # calculate likelihood using gaussian distribution
    # each pixel is row of X
    constant_in_likelihood = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov)))
    sigma_inv = np.linalg.inv(cov)
    X2 = X - mean
    exponent = (-0.5) * (np.dot(X2, sigma_inv) * X2).sum(1)
    likelihood = constant_in_likelihood * np.exp(exponent)

    # posterior
    prior = 0.5
    posterior = prior * likelihood

    # mask
    mask = posterior
    mask[mask < tau] = -1  # temperarily mark as -1 to prevent being marked as 0 later
    mask[mask != -1] = 0
    mask[mask == -1] = 1
    mask = mask.reshape(l, w)  # reshape back to 2D image
    # print("end vectorized algorithm", datetime.now().strftime("%H:%M:%S"))

    ##  show mask
    three_d_mask = np.stack((mask, mask, mask), axis=2)
    masked_img = np.multiply(three_d_mask, img)
    image_name = os.path.join(output_dir, "masked_" + str(img_name))
    #cv2.imshow(image_name, masked_img)
    cv2.imwrite(os.path.join(output_dir,"single_gaussian_" + str(img_name)), masked_img)
    cv2.waitKey(0)
    print("Finish Generating mask for image ", str(img_name))


if __name__ == "__main__":
    # load data
    input_dir = "train_images"  # path to the train image dataset
    # output directory
    if not (os.path.isdir("single_gaussian_result")):
        os.mkdir("single_gaussian_result")
    output_dir = "single_gaussian_result"
    print(" Single gussian result are stored in the directory called single_guassian_result")
    # User defined threshold
    tau = 0.00000000000000001
    # Number of process
    print("Starting #", mp.cpu_count(), "of process")
    pool = mp.Pool()
    for img in os.listdir(input_dir):
        pool.apply_async(func=single_gussian, args=(img, input_dir, output_dir, tau,))
    pool.close()
    pool.join()
    print("All Process are finished")