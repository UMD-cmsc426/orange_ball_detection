import math
import numpy as np


def testGMM(K, threshold, scaling, mean, cov, img):
    pixels = []
    for w in range(len(img[:, 0, 0])):
        for h in range(len(img[0, :, 0])):
            pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
            # compute posterior for the pixel
            posterior = 0
            for i in range(K):
                likelihood = get_likelihood(pix, mean, cov)
                posterior += scaling * likelihood
            # classify
            if posterior >= threshold:
                pixels.append(pix)

    return pixels


def get_likelihood(pixel, mean, cov):
    return math.exp((pixel - mean).T.dot(np.linalg.inv(cov)).dot(pixel - mean)) / math.sqrt(
        pow(2 * math.pi, 3) * np.linalg.det(cov))
