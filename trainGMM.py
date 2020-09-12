import testGMM
import numpy as np


def trainGMM(K, img, max_iter):
    converge_criteria = 0
    # TODO: randomly initialize parameters for all clusters
    params = []
    for i in range(K):
        scaling = 0
        mean = 0
        cov = []
        params.append((scaling, mean, cov))

    sum_means = 0
    sum_prev_means = 0

    while i <= max_iter and abs(sum_means - sum_prev_means) > converge_criteria:
        # TODO: define sum_means and sum_prev_means
        # Expectation step - assign points to clusters, get cluster weight
        weights = []
        for cluster in range(K):
            cluster_weights = []
            cluster_scaling, cluster_mean, cluster_cov = params[cluster]

            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    likelihood = testGMM.get_likelihood(pix, cluster_mean, cluster_cov)
                    denom = 0

                    for k in range(K):
                        k_scaling, k_mean, k_cov = params[cluster]
                        k_likelihood = testGMM.get_likelihood(pix, k_mean, k_cov)
                        denom += k_scaling * k_likelihood

                    weight = cluster_scaling * likelihood / denom
                    cluster_weights.append(weight) # probability of each pixel belonging to this cluster
            weights.append(cluster_weights) # weights for all clusters 1 to K,
            # weights[i][j]is the probability of the jth pixel belonging to the ith cluster

        # Maximization step - get new scaling, mean, and cov for each cluster
        for i in range(K):
            j = 0   # pixel index
            new_scaling = 0
            new_cov_num = []
            new_mean_nom = 0
            sum_cluster_weights = 0

            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    new_mean_nom += weights[i][j] * pix
                    sum_cluster_weights += weights[i][j]
                    j += 1
            new_mean = new_mean_nom / sum_cluster_weights
            new_scaling = sum_cluster_weights / (w*h)

            for w in range(len(img[:, 0, 0])):
                for h in range(len(img[0, :, 0])):
                    pix = np.asmatrix([[img[w][h][0]], [img[w][h][1]], [img[w][h][2]]])
                    new_cov_num += weights[i][j] * (pix - new_mean).dot((pix - new_mean).T)
            new_cov = new_cov_num / sum_cluster_weights

            # update model
            params[i] = (new_scaling, new_mean, new_cov)

        i += 1








