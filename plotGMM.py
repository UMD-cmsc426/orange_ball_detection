import numpy as np
import re
import cv2
import os
import random
from gaussian import *
from trainGMM import *
from testGMM import *


def plotGMM(params):
    for i in range(len(params)):
        scaling, mean, cov = params[i]
        # mean[0][0], mean[1][0], mean[2][0], cov[0,0], cov[1,1], cov[2,2]
        # 咋画呢？




    pass

if __name__ == "__main__":
    try:
        # try to load weights. If fials, run training.
        with open(os.path.join(output_dir, "weights"), "wb") as f:
            params = np.load(f)
    except:
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

        with open(os.path.join(output_dir, "weights"), "wb") as f:
            np.save(f,params)

    plotGMM(params)
