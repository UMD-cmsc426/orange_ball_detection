# import packages
import cv2
import os
import numpy as np
import math
import multiprocessing
import time

# Define a function that generates a RGB image's mean value and covariance.
# Output shape: mean-（3x1） covariacne-(3x3)
def cal_mean_cov(img):
    l, w, h = img.shape
    mean = [[np.sum(img[:,:,0])/(l*w)],[np.sum(img[:,:,1])/(l*w)],[np.sum(img[:,:,2])/(l*w)]]
    cov = np.zeros((3,3),)
    #R_value = []
    for width in range(len(img[:,0,0])):
        for length in range(len(img[0,:,0])):
            RGB_value = [[img[width][length][0]],[img[width][length][1]],[img[width][length][2]]]
            cov = cov + (np.asmatrix(RGB_value) - np.asmatrix(mean))@(np.asmatrix(RGB_value) - np.asmatrix(mean)).T
    cov = cov/(l*w)
    return mean,cov

def single_gussian(img_name,input_dir,output_dir, tau):
    img = cv2.imread(os.path.join(input_dir,img_name))
    mean, cov = cal_mean_cov(img)
    # creat a mask to indicate the position of ball
    mask = np.zeros((img.shape[:-1]))
    # print("cov: \n", cov)
    prior = 0.5
    likelihood_list = []
    for width in range(len(img[:, 0, 0])):
        for length in range(len(img[0, :, 0])):
            curr_pixel = np.asmatrix([[img[width][length][0]], [img[width][length][1]], [img[width][length][2]]])
            likelihood = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov))) * math.exp(
                (-0.5) * (curr_pixel - mean).T @ np.linalg.inv(cov) @ (curr_pixel - mean))
            # print("likelihood", likelihood)
            likelihood_list.append(likelihood)
            if (likelihood * prior < tau):
                mask[width][length] = 1
    # stack mask to three channels  inorder to produce masks
    three_d_mask = np.stack((mask, mask, mask), axis=2)
    masked_img = np.multiply(three_d_mask, img)
    image_name = os.path.join(output_dir,"masked_" + str(img_name))
    print(image_name)
    # store image to output directory
    cv2.imwrite(image_name, masked_img)
    cv2.waitKey()
    print("Finish Generating mask for image ", str(img_name))

if __name__ == "__main__":
    # load data
    input_dir = "train_images"  # path to the train image dataset
    # output directory
    output_dir = "single_gaussian_result"
    # User defined threshold
    tau = 0.00000000000000001
    # Number of process
    num_process = 16
    jobs = []
    for img in os.listdir(input_dir):
       # p = multiprocessing.Process(target=mask_generator(os.path.join(input_dir,img), output_dir, tau))
        p = multiprocessing.Process(target=single_gussian ,args=(img, input_dir,output_dir, tau,))
        while (len(jobs) > num_process):
            time.sleep(5)
        jobs.append(p)
        print("Starting a Process")
        p.start()

    print("All Process are finished")