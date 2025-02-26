from dataclasses import dataclass

import numpy as np

from utils import binary_search_boundaries


def get_cdf_boundaries(tagged_points, m):
    n = len(tagged_points)

    if n % m != 0:
        raise ValueError("Number of points must be divisible by m")

    points = map(lambda p: p[0], tagged_points)

    # Sort points based on first attribute
    x_values = sorted(map(lambda p: p[0], points))

    return [(x_values[j * (n // m) - 1] + x_values[j * (n // m)]) / 2 for j in range(1, m)]


@dataclass
class CDFBasedHashFunction:
    boundaries: list[float]

    def __call__(self, point: np.ndarray) -> int:
        return binary_search_boundaries(self.boundaries, point[0])



def create_hash_function(points, buckets_amount):
    boundaries = get_cdf_boundaries(points, buckets_amount)
    return CDFBasedHashFunction(boundaries)
