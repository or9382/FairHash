from dataclasses import dataclass
from math import floor

import numpy as np

from utils import binary_search_boundaries


def seep_and_cut(tagged_points, m, group_lengths):
    n = len(tagged_points)
    k = len(group_lengths)

    if n % m != 0:
        raise ValueError("Number of points must be divisible by m")

    # Sort points based on first attribute
    tagged_points = sorted(tagged_points, key=lambda tagged_point: tagged_point[0][0])

    c = [0] * k
    H_temp = []
    for i in range(n):
        p, j = tagged_points[i]
        H_temp.append(floor((c[j] * m) / group_lengths[j]))
        c[j] += 1

    i = 0
    H = []
    B = []
    while True:
        while i < n - 1 and H_temp[i] == H_temp[i+1]:
            i += 1

        H.append(H_temp[i])
        if i == n - 1:
            break

        B.append((tagged_points[i][0][0] + tagged_points[i + 1][0][0]) / 2)
        i += 1

    return B, H


@dataclass
class CutBasedHashFunction:
    boundaries: list[float]
    cut_to_bucket: list[int]

    def __call__(self, point: np.ndarray) -> int:
        split = binary_search_boundaries(self.boundaries, point[0])
        return self.cut_to_bucket[split]


def create_hash_function(points, buckets_amount, groups):
    B, H = seep_and_cut(points, buckets_amount, groups)
    return CutBasedHashFunction(B, H)
