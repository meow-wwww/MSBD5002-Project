# Author: LUO Zixi
import numpy as np

# constraint of non-negative H >= 0^2
def matrixFacNonNegative(M, K, H):
    count = 0
    while count < K:
        grad = 4 * (np.dot(H, np.dot(np.transpose(H), H)) - np.dot(M, H))
        step = 1 / (2*(4 * (np.linalg.norm(np.dot(H, np.transpose(H)) - M)) + 8 * np.linalg.norm(np.dot(np.transpose(H), H))))
        h = H - step * grad
        h[h < 0] = 0
        count += 1
    return H

# constraint of 2-norm of h is 1
def matrixFacUnit(M, K, H):
    count = 0
    while count < K:
        grad = 4 * (np.dot(H, np.dot(np.transpose(H), H)) - np.dot(M, H))
        step = 1 / (2 * (4 * (np.linalg.norm(np.dot(H, np.transpose(H)) - M)) + 8 * np.linalg.norm(np.dot(np.transpose(H), H))))
        h = (H - step * grad) / np.linalg.norm(H - step * grad)
        count += 1
    return h
