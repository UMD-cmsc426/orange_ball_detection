from trainGMM import *
from testGMM import *
from measureDepth import *
import os
import cv2
import numpy as np
import random
from gaussian import *


train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset
# output directory
output_dir = "results"


def gmm():
    # User defined threshold
    tau_train = 0.7
    tau_test = 0.0000004
    prior = 0.5
    K = 20
    max_iter = 500
    clusters = []
    depths = []

    
    # load training data
    X = extract_orange_pixels()
    # train
    params = trainGMM(K, max_iter, X, tau_train)
    print("Finish Training")
    
    # Load test images
    test_images = load_images("test_images")

    # test
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    cluster, mask, depth = testGMM(params, tau_test, K, prior)

    # TODO: measure depth and plot GMM

    return clusters, depth


def load_images(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            images.append((img, file_name))
    return images


if __name__ == "__main__":
    gmm()
    print("All images have been processed. Press any key on images to exit.")
    cv2.waitKey(0)
