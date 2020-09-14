import trainGMM
import testGMM
import measureDepth
import cv2
import os
import numpy as np


def gmm(training):
    # Set threshold and number of gaussians
    threshold = 0.0001
    K = 5
    clusters = []
    depths = []

    if training:
        # Load training images
        train_images = load_images("train_images")
        for train_img, img_name in train_images:
            trainGMM.trainGMM(K, 100, train_img, img_name)
    else:
        # Load test images
        test_images = load_images("test_images")
        for test_img, img_name in test_images:
            cluster, mask = testGMM.testGMM(K, threshold, test_img, img_name)
            clusters.append(cluster)
            depth = measureDepth.measureDepth(cluster, test_img)
            depths.append(depth)

            # produce masked image
            three_d_mask = np.stack((mask, mask, mask), axis=2)
            masked_img = np.multiply(three_d_mask, test_img)
            image_name = os.path.join("GMM_result", "masked_" + str(img_name))
            # store image to output directory
            cv2.imwrite(image_name, masked_img)
            cv2.waitKey()

    # TODO: plotGMM

    return clusters, depth


def load_images(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            images.append((img, file_name))
    return images


if __name__ == "__main__":
    gmm(True)
