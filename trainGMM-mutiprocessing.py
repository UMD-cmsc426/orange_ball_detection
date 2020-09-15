import os
import cv2
import numpy as np
import random
import sys
import testGMM
import re
import copy
import multiprocessing as mp
import time


def calculate_likelihood(w, img, cluster_mean, cluster_cov, cluster_scaling, shared:mp.Queue):
    tem_weight = np.zeros(len(img[0, :, 0]))
    # print("shape of tem weights ", tem_weight.shape)
    for h in range(len(img[0, :, 0])):
        pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
        #print('[in process] row {} started!'.format(h))
        try:
            likelihood = min(max(testGMM.get_likelihood(pix, cluster_mean, cluster_cov), 1e-40), 1.e40)
        except:
            likelihood = 1e-40
        tem_weight[h] = (likelihood * cluster_scaling)

    cumulated_weights, cluster_weights = shared.get()
    cluster_weights[w] = [tem_weight][0]
    cumulated_weights[w] += [tem_weight][0]
    shared.put([cumulated_weights, cluster_weights])
    # print('i released it!')
    sys.stdout.flush()

# parameters:
# k: int, number of guassian distribution
# max_iter: int, maximum number of step in optimization
# img_name: strings, the relative path to single image, i.e. "train_images/032.jpg"
def trainGMM(K, max_iter, img_name):
    global pool
    print('train GMM!')

    # read img
    img = cv2.imread(img_name)
    # user defined converge threshold
    tau = 0.00000000000000001

    def initialize():
        mean = np.asmatrix([[random.randint(1, 255)], [random.randint(1, 255)], [random.randint(1, 255)]])
        # generate a random positive-semidefinete matrix as covariance matrix
        A = np.random.random((3, 3)) * 20
        cov = np.dot(A, A.transpose())
        scaling = (random.random() * 3.0)
        return [scaling, mean, cov]

    params = [initialize() for cluster in range(K)]
    print('init complete!')
    # Structure of para:
    # [[scale,mean,covariance],[scale,mean,covariance],[scale,mean,covariance]...]
    # scale is a int. Mean is a 3x1 matrix. Covariance is a 3x3 matrix

    # total_mean is the sum of all mean from different clusters
    total_mean = np.full((K, 3, 3), -9999)
    prev_total_mean = np.full((K, 3, 3), 9999)
    iter = 0

    # return true if MLE converges. Return false otherwise
    def check_convergence(total_mean, prev_toal_mean, tau):
        sum = 0
        print("For cluster 1, total_mean and Prev_total_mean are as follow\n")
        print("curr mean: \n", total_mean[0])
        print("prev mean: \n", prev_total_mean[0])
        for cluster in range(len(prev_toal_mean)):
            sum += np.linalg.norm(total_mean[cluster] - prev_total_mean[cluster])
        print("Check convergence difference: ", sum)
        return sum >= tau

    while iter < max_iter and check_convergence(total_mean, prev_total_mean, tau):
        print("iter: ", iter)
        # update prev total mean
        prev_total_mean = copy.deepcopy(total_mean)

        # Expectation step - assign points to clusters, get cluster weight
        weights = np.asarray([np.zeros((img.shape[0], img.shape[1])) for _ in range(K)])
        for cluster in range(K):
            print('cluster =', cluster)
            # weight for a single cluster
            cluster_weights = np.zeros((img.shape[0], img.shape[1]))
            # cumulated weights add up all weights on a given pixel -- serving as denominator
            cumulated_weights = np.zeros((img.shape[0], img.shape[1]))
            cluster_scaling, cluster_mean, cluster_cov = params[cluster]

            # start multiprocessing
            shared = mp.Manager().Queue()
            shared.put([cumulated_weights, cluster_weights])
            pool = mp.Pool(mp.cpu_count() * 2)
            for w in range(len(img[:, 0, 0])):
                pool.apply_async(calculate_likelihood,
                                 (w, img, cluster_mean, cluster_cov, cluster_scaling, shared))
                # pool.apply_async(calculate_likelihood, args=(
                #         w, img, cluster_mean, cluster_cov, cluster_scaling, shared,))
            pool.close()
            pool.join()
            # end of multiprocessing
            [cumulated_weights, cluster_weights] = shared.get()
            weights[cluster] = cluster_weights  # weights for all clusters 1 to K,
            if (weights[cluster] == 0).all():
                raise Exception("all weights are zero")
            if (weights[cluster] == 0).any():
                raise Exception("one of the weights is zero")

        for cluster in range(K):
            print('cluster2 =', cluster)
            weights[cluster] = np.divide(np.double(weights[cluster]), np.double(cumulated_weights))

        # ---- Another multiprocessing start here
        # Maximization step - get new scaling, mean, and cov for each cluster
        for cluster in range(K):
            mean_sum = np.zeros((3, 1))  # sums all weight*pixel RGB value on image
            cov_sum = np.zeros((3, 3))
            sum_weights = np.sum(weights[cluster])  # sum of all the weights given a cluster
            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    # calculate mean
                    mean_sum += np.multiply(weights[cluster][w][h], pix)

            new_mean = np.divide(np.double(mean_sum), np.double(sum_weights))
            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    # calculate covariance
                    cov_sum += np.multiply(weights[cluster][w][h], (pix - new_mean)) @ ((pix - new_mean).T)
            new_cov = np.divide(np.double(cov_sum), np.double(sum_weights))
            new_scaling = sum_weights / (img.shape[0] * img.shape[1])
            mean_sum += mean_sum

            total_mean[cluster] = new_mean
            print("new mean at cluster ", cluster, "is \n", new_mean)
            # update model
            params[cluster] = (new_scaling, new_mean, new_cov)
            # ---- end of mutiprocessing here; join here
        print("-----------------")
        iter += 1

    # store weights to .npy
    if not os.path.exists("weights"):
        os.mkdir("weights")
    else:
        digit = re.findall(r'\d+\d+\d*', img_name)
        file_name = str(digit[0]) + "_weight.npy"
        with open(os.path.join("weights", file_name), "wb") as f:
            np.save(f, params)


if __name__ == "__main__":
    #np.seterr(all='raise')
    input_dir = "train_images"
    for img_name in os.listdir(input_dir):
        img = os.path.join(input_dir, img_name)
        print('img:', img)
        trainGMM(5, 200, img)
        print("Finish Training for ", img)
        break
