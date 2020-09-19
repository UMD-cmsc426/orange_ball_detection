import math
import numpy as np
def get_likelihood(pixel, mean, cov):
    result = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov))) * math.exp((-0.5) * (pixel - mean).T @ np.linalg.inv(cov) @ (pixel - mean))
    return  result
def test():
    A = np.asarray([[[1,2,3], [4, 5, 6], [7, 8, 9]],[[10,11,12], [14, 15, 16], [17, 18, 19]],[[21,22,23], [24, 25, 26], [72, 82, 92]]])
    B = np.asarray([[0],[0],[0]])
    print("A: \n", A)
    print("B: \n", B)
    print("\n\n")
    def print_RGB(A, B):
        return
if __name__ == "__main__":
    test()