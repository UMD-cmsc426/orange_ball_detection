import trainGMM
import testGMM
import measureDepth
import cv2
import os


def gmm(training):
    # Set threshold and number of gaussians
    threshold = 0.0001
    K = 5
    clusters = []
    depths = []

    if training:
        # Load training images
        train_images = load_images("train_images")
        K, scaling, mean, cov = trainGMM.trainGMM(K, train_images, max_iter=100)
    else:
        # Load test images
        test_images = load_images("test_images")
        for test_img in test_images:
            cluster = testGMM.testGMM(K, threshold, test_img)
            clusters.append(cluster)
            depth = measureDepth.measureDepth(cluster, test_img)
            depths.append(depth)
    plotGMM()

    return clusters, depth


def load_images(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            images.append(img)
    return images
