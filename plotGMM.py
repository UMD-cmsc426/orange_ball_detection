import numpy as np
import re
import cv2
import os


def plotGMM(img_name):
    digit = re.findall(r'\d+\d+\d*', img_name)
    file_name = str(digit[0]) + "_weight.npy"
    with open(os.path.join("weights", file_name), "wb") as f:
        params = np.load(f)

    for i in range(len(params)):
        scaling, mean, cov = params[i]
        # mean[0][0], mean[1][0], mean[2][0], cov[0,0], cov[1,1], cov[2,2]
        # 咋画呢？




    pass
