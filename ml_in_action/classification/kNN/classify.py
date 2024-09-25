import numpy as np
from numpy.typing import NDArray

def distance(p1:NDArray, p2:NDArray) -> np.float64:
    dim = max(np.size(p1), np.size(p2))
    if np.size(p1) < dim :
        p1 = np.pad(p1, dim-np.size(p1), mode='constant', constant_values=0)
    if np.size(p2) < dim :
        p2 = np.pad(p2, dim-np.size(p2), mode='constant', constant_values=0)
    vec = np.array(
        [p1[i] - p2[i] for i in range(dim)],
        dtype=np.float64
    )
    dist_2 = np.array(
        [compent ** 2 for compent in vec],
        dtype=np.float64
    ).sum(axis=0)
    dist = np.sqrt(dist_2)
    return dist

def distances(p1:NDArray, p2s:list[NDArray]) -> NDArray:
    return np.array(
        [distance(p1, p2) for p2 in p2s],
        dtype=np.float64
    )

def classify(inX, group, features, k:int):
    k_indicies = (distances(inX, group).argsort())[:k]
    class_count = {}
    for k_index in k_indicies:
        k_feature = features[k_index]
        class_count[k_feature] = class_count.get(k_feature, 0) + 1
    
    return sorted(
        class_count.items(),
        key=(lambda item: item[1]),
        reverse=True
    )[0][0]