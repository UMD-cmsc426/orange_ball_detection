import math
import numpy as np
def get_likelihood(pixel, mean, cov):
    result = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov))) * math.exp((-0.5) * (pixel - mean).T @ np.linalg.inv(cov) @ (pixel - mean))
    return  result
def expoent_vectorized(mean_diff, sigma_inv):
    return (-0.5) * (np.dot(mean_diff.T, sigma_inv) * mean_diff.T)
def test():
    A = np.asarray([[[1,2,3], [4, 5, 6], [7, 8, 9]],[[10,11,12], [14, 15, 16], [17, 18, 19]],[[21,22,23], [24, 25, 26], [72, 82, 92]]])
    B = np.asarray([[0],[0],[0]])
    sigma_inv = np.asarray([[1.00845842, - 2.05290203,  0.54121026],[-2.05290203,  4.34896318, - 1.24044691],[0.54121026, - 1.24044691,0.40686255]])
    print("A: \n", A)
    print("B: \n", B)
    print("\n\n")

    def my_function_allong_axix(M, argument):
        return np.apply_along_axis(expoent_vectorized, 0, M, argument)
    print(my_function_allong_axix(A, sigma_inv))
if __name__ == "__main__":
    test()