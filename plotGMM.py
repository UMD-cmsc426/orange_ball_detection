from testGMM import *
import matplotlib.pyplot as plt
# load data
train_dir = "train_images"# path to the train image dataset
test_dir = "test_images"# path to the train image dataset
# output directory
output_dir = "results"


def plotGMM(params):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for param in params:
        scaling, mean, cov = param
        w, v = np.linalg.eig(cov)
        # Define ellipsoid center
        x_r, y_r, z_r = scaling * w
        # calculate zenith angle
        x_vec, y_vec, z_vec = v.T
        theta = np.arccos(np.clip(np.dot(x_vec, y_vec), -1.0, 1.0))
        phi = np.arccos(np.clip(np.dot(z_vec, [0, 0, 1]), -1.0, 1.0))
        # populate theta and phi
        phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
        theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle

        x = x_r * np.sin(theta) * np.cos(phi)
        y = y_r * np.sin(theta) * np.sin(phi)
        z = z_r * np.cos(theta)
        ax.plot_surface(x, y, z)
    plt.show
    plt_name = os.path.join(output_dir, "ellipsoid", "ellipsoid_plot")
    if not (os.path.isdir(os.path.join(output_dir, "ellipsoid"))):
        os.mkdir(os.path.join(output_dir, "ellipsoid"))
    fig.savefig(fname=plt_name, dpi=fig.dpi)
    print("Ellipsoid has been saved at /result/ellipsoid/ellipsoid_plot.png")

