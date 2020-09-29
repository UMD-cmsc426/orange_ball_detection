import cv2
import os
import re
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


def measure_depth_train():
    GMM_dir = os.path.join("results", "GMM")

    # GET a list of distance
    dis_list = []
    num_list = []

    for img_name in os.listdir(GMM_dir):
        dis = int(re.search(r"GMM_([0-9]+)\.jpg",img_name)[1])
        mask = cv2.imread(os.path.join(GMM_dir, img_name))
        num_pixel = cv2.countNonZero(cv2.inRange(mask, (20, 20, 20),(240, 240, 240) ))
        dis_list.append(dis)
        num_list.append(num_pixel)
    train_list = list(zip(num_list, dis_list))
    train_list.sort()

    params, _ = optimize.curve_fit(inverse_square, *zip(*train_list))

    print("params: ", params)
    plt.scatter(*zip(*train_list), label='Data')
    plt.plot(list(zip(*train_list))[0], inverse_square(list(zip(*train_list))[0], params[0], params[1]),
             label='Fitted function')
    plt.legend(loc='best')
    plt.show()

    return params


def measure_depth_predict(params):
    # Predict distance for test set images
    test_dir = os.path.join("results", "GMM_test")
    output_dir = os.path.join("results", "distances")
    for img_name in os.listdir(test_dir):
        mask = cv2.imread(os.path.join(test_dir, img_name))
        num_pixel = cv2.countNonZero(cv2.inRange(mask, (20, 20, 20), (240, 240, 240)))
        predicted_dist = inverse_square(num_pixel, *params)
        print(predicted_dist)
        # Output test images with predicted distance
        h, w, _ = mask.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(mask, "D = {}".format(predicted_dist), (int(h / 5), int(w / 5)), font, 0.5, (225, 225, 225),
                          1)
        img_name = os.path.join(output_dir, "Distance_" + str(img_name))
        cv2.imwrite(img_name, img)


def inverse_square(x, a, b):
    # Fit function for area vs. distance
    return np.sqrt(a/b/x)
