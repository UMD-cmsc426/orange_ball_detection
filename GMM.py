from trainGMM import *
from plotGMM import *
from testGMM import *
from measureDepth import *
train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset
# output directory
train_output_dir = os.path.join("results", "depth_train")
test_output_dir = os.path.join("results", "GMM_test")

def gmm(tau_train, tau_test, prior, K, max_iter, training = True):
    clusters = []
    depth = []
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    if training:
        # load training data
        X = extract_orange_pixels()
        # train
        print("--- Start Training ---")
        params = trainGMM(K, max_iter, X, tau_train)
        with open(os.path.join(output_dir, "weights"), "wb") as f:
            np.save(f, params, allow_pickle=True)
    else:
        # testing:
        print("Here: ", os.path.isfile(os.path.join(output_dir, "weights")))
        try:
            with open(os.path.join(output_dir, "weights"), "rb") as f:
                params = np.load(f, allow_pickle=True)
        except Exception:
            raise Exception("No training Model found! Please train first")
        print("--- Start Testing ---")
        testGMM(params, tau_test, K, prior, train_dir, train_output_dir)
        testGMM(params, tau_test, K, prior, test_dir, test_output_dir)
        # measure depth
        depth_params = measure_depth_train()
        measure_depth_predict(depth_params)

    # plot GMM
    plotGMM(params)
    print("--- END ---")
    return clusters, depth


if __name__ == "__main__":
    # User defined threshold
    tau_train = 0.5
    tau_test = 0.0000004
    prior = 0.2
    K = 20
    max_iter = 500
    if not (os.path.isdir(train_output_dir)):
        os.mkdir(train_output_dir)
    if not (os.path.isdir(test_output_dir)):
        os.mkdir(test_output_dir)
    gmm(tau_train, tau_test, prior, K, max_iter, training=True)
    gmm(tau_train, tau_test, prior, K, max_iter, training=False)
