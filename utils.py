import numpy as np


# Calculate the projection of a point onto a vector
def calculate_projection(vector: np.ndarray, point: np.ndarray):
    return np.dot(vector, point) / np.linalg.norm(vector)


# Calculate the angle from the positive x-axis for the given vector
def calculate_angle(p: np.ndarray):
    xp, yp = p

    # Calculate the mathematical angle in radians
    return np.arctan2(yp, xp)


# Finds the set and element indices where the point `p` is located.
def find_set_index(sets, p) -> int:
    for set_index, s in enumerate(sets):
        try:
            element_index = next(j for j, q in enumerate(s) if np.allclose(p, q[0]))
        except StopIteration:
            continue

        return set_index, element_index
    else:
        raise ValueError(f"Point {p} not found in any set")


# Returns a vector that a slightly smaller angle than the given vector
def get_vector_with_slightly_smaller_angle(vector: np.ndarray):
    angle = calculate_angle(vector) - 1e-6  # Slightly decrease the angle
    magnitude = np.linalg.norm(vector)  # Preserve the magnitude of the vector
    return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle)])


# Returns a vector that a slightly bigger angle than the given vector
def get_vector_with_slightly_bigger_angle(vector: np.ndarray):
    angle = calculate_angle(vector) + 1e-6  # Slightly increase the angle
    magnitude = np.linalg.norm(vector)  # Preserve the magnitude of the vector
    return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle)])


def binary_search_boundaries(boundaries, value):
    low, high = 0, len(boundaries) - 1
    while low <= high:
        mid = (low + high) // 2
        if boundaries[mid] < value:
            low = mid + 1
        else:
            high = mid - 1
    return low


# Build array of amount of intersections between buckets and groups
def get_bucket_group_matrix(buckets, group_lengths):
    m = len(buckets)
    k = len(group_lengths)

    return [[len([0 for (p, t) in buckets[j] if t == i]) for j in range(m)] for i in range(k)]


def calculate_pairwise_group_fairness(group_in_bucket, group_size):
    return (group_in_bucket / group_size) ** 2


def calculate_pairwise_fairness(group_bucket_arr, group_size):
    return sum(calculate_pairwise_group_fairness(group_in_bucket, group_size) for group_in_bucket in group_bucket_arr)


def calculate_pairwise_fairness_list(bucket_group_matrix, group_lengths):
    k = len(group_lengths)

    return [calculate_pairwise_fairness(bucket_group_matrix[i], group_lengths[i]) for i in range(k)]


def calculate_epsilon(pairwise_fairness, bucket_count):
    return bucket_count * max(pairwise_fairness) - 1
