import math
import numpy as np
def get_likelihood(pixel, mean, cov, queue = None):
    result = 1 / (math.sqrt(((2 * math.pi) ** 3) * np.linalg.det(cov))) * math.exp((-0.5) * (pixel - mean).T @ np.linalg.inv(cov) @ (pixel - mean))
    if queue != None:
        queue.put(result)
    return  result