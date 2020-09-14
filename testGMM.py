import math
import numpy as np
import sys
import re
import os

def testGMM(K, threshold, img):
    digit = re.findall(r'\d+\d+\d*', img)
    file_name = str(digit[0]) + "_weight.npy"
    with open(os.path.join("weights", file_name), "wb") as f:
        params = np.load(f)

    pixels = []
    for w in range(len(img[:, 0, 0])):
        for h in range(len(img[0, :, 0])):
            pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
            # compute posterior for the pixel
            posterior = 0
            for i in range(K):
                scaling, mean, cov = params[i]
                likelihood = get_likelihood(pix, mean, cov)
                posterior += scaling * likelihood
            # classify
            if posterior >= threshold:
                pixels.append(pix)

    return pixels


def get_likelihood(pixel, mean, cov):
    if np.all(pixel == 0) or np.all(pixel == 0) or np.all(pixel == 0):
        print("pixel\n ", pixel)
        print("mean\n ", mean)
        print("cov\n ", cov)
        sys.exit()
    # math.exp((pixel - mean).T.dot(np.linalg.inv(cov)).dot(pixel - mean)) / math.sqrt(pow(2 * math.pi, 3) * np.linalg.det(cov))
    return 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov))) * math.exp((-0.5) * (pixel - mean).T @ np.linalg.inv(cov) @ (pixel - mean))
