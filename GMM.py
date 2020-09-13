import trainGMM, testGMM, measureDepth,plotGMM
import cv2
import os

def gmm(training = True):
    # Set threshold and number of gaussians
    threshold = 0.0001
    K = 5
    clusters = []
    depths = []

    # It's a little bit inefficient to train before test.
    # Load training images
    train_images = load_images("train_images")
    K, scaling, mean, cov = trainGMM.trainGMM(K, train_images, max_iter=100)
    if not training:
        # Load test images
        test_images = load_images("test_images")
        for test_img in test_images:
            cluster = testGMM.testGMM(K, threshold, scaling, mean, cov, test_img)
            clusters.append(cluster) ### not sure here
            depth = measureDepth.measureDepth(cluster, test_img)
            depths.append(depth)
            plotGMM(scaling, mean, cov, test_img)

    return clusters, depths # why return a batch?



def load_images(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            images.append(img)
    return images





