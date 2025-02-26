from dataclasses import dataclass
from itertools import combinations

from heapdict import heapdict
import numpy as np

from utils import calculate_projection, find_set_index, get_vector_with_slightly_smaller_angle, \
    binary_search_boundaries, get_bucket_group_matrix, calculate_pairwise_group_fairness, calculate_pairwise_fairness, \
    calculate_epsilon, get_vector_with_slightly_bigger_angle


def compute_linear_ranking_functions(tagged_points):
    """
    Computes all possible linear ranking functions that order the points based on projection.
    The functions are ordered such that between each consecutive function, only two adjacent points swap ranks.
    Returns a list of tuples: (a, b) representing the function ax + by = 1.
    """
    intersections = []

    # Consider all pairs of points
    for (p1, t1), (p2, t2) in combinations(tagged_points, 2):

        # Solve for (a, b) in the equation a*x + b*y = 1
        A = np.array([p1, p2])
        B = np.array([1, 1])

        try:
            vector = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue  # Skip if singular matrix (parallel lines)

        vector /= np.linalg.norm(vector)

        smaller_angle = get_vector_with_slightly_smaller_angle(vector)
        if calculate_projection(smaller_angle, p1) < calculate_projection(smaller_angle, p2):
            intersections.append(((p1, t1), (p2, t2), vector))
        else:
            intersections.append(((p2, t2), (p1, t1), vector))

    # Sort by angle to ensure a consistent order of swaps
    intersections.sort(key=lambda x: np.arctan2(x[2][1], x[2][0]))
    return intersections


def get_best_ranking(tagged_points, m, group_lengths):
    n = len(tagged_points)
    k = len(group_lengths)

    if n % m != 0:
        raise ValueError("Number of points must be divisible by m")

    W = compute_linear_ranking_functions(tagged_points)
    w0 = get_vector_with_slightly_smaller_angle(W[0][2])

    Pw = sorted(tagged_points, key=lambda p: calculate_projection(w0, p[0]))
    buckets = [list(Pw[j * (n // m):(j + 1) * (n // m)]) for j in range(m)]

    a = get_bucket_group_matrix(buckets, group_lengths)

    Pr = [calculate_pairwise_fairness(a[i], group_lengths[i]) for i in range(k)]
    M = heapdict({i: -p for i, p in enumerate(Pr)})
    epsilon = calculate_epsilon(Pr, m)

    best_w = w0
    for (ps, p_group), (qs, q_group), ws in W:
        p_bucket, p_idx_in_bucket = find_set_index(buckets, ps)
        q_bucket, q_idx_in_bucket = find_set_index(buckets, qs)

        buckets[p_bucket][p_idx_in_bucket] = (qs, q_group)
        buckets[q_bucket][q_idx_in_bucket] = (ps, p_group)

        if p_bucket != q_bucket and p_group != q_group:
            Pr[p_group] = Pr[p_group] - \
                          calculate_pairwise_group_fairness(a[p_group][p_bucket],     group_lengths[p_group]) - \
                          calculate_pairwise_group_fairness(a[p_group][q_bucket],     group_lengths[p_group]) + \
                          calculate_pairwise_group_fairness(a[p_group][p_bucket] - 1, group_lengths[p_group]) + \
                          calculate_pairwise_group_fairness(a[p_group][q_bucket] + 1, group_lengths[p_group])

            Pr[q_group] = Pr[q_group] - \
                          calculate_pairwise_group_fairness(a[q_group][q_bucket],     group_lengths[q_group]) - \
                          calculate_pairwise_group_fairness(a[q_group][p_bucket],     group_lengths[q_group]) + \
                          calculate_pairwise_group_fairness(a[q_group][q_bucket] - 1, group_lengths[q_group]) + \
                          calculate_pairwise_group_fairness(a[q_group][p_bucket] + 1, group_lengths[q_group])

            a[p_group][p_bucket] -= 1
            a[p_group][q_bucket] += 1
            a[q_group][q_bucket] -= 1
            a[q_group][p_bucket] += 1

            M[p_group] = -Pr[p_group]
            M[q_group] = -Pr[q_group]

            M_top = -M.peekitem()[1]
            if m * M_top - 1 < epsilon:
                epsilon = m * M_top - 1
                best_w = ws

    best_projection = sorted(map(lambda p: calculate_projection(best_w, p[0]), tagged_points))
    B = [(best_projection[j * (n // m) - 1] + best_projection[j * (n // m)]) / 2 for j in range(1, m)]

    return best_w, B


@dataclass
class RankBasedHashFunction:
    vector: np.ndarray
    boundaries: list[float]

    def __call__(self, point: np.ndarray) -> int:
        projection = calculate_projection(self.vector, point)
        return binary_search_boundaries(self.boundaries, projection)


def create_hash_function(points, buckets_amount, groups):
    vector, boundaries = get_best_ranking(points, buckets_amount, groups)
    return RankBasedHashFunction(vector, boundaries)


def main():
    pass


if __name__ == '__main__':
    main()
