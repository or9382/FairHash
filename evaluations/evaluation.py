import time

from utils import get_bucket_group_matrix, calculate_pairwise_fairness_list, calculate_epsilon


def measure_time(function, *args, **kwargs):
    start_time = time.perf_counter()
    res = function(*args, **kwargs)
    end_time = time.perf_counter()

    return res, end_time - start_time


def test(hash_function, points, buckets_amount, group_lengths):
    buckets = {i: [] for i in range(buckets_amount)}

    def query():
        for p in points:
            buckets[hash_function(p[0])].append(p)

    _, query_time = measure_time(query)

    a = get_bucket_group_matrix(list(list(bucket) for bucket in buckets.values()), group_lengths)
    fairness = calculate_pairwise_fairness_list(a, group_lengths)
    epsilon = calculate_epsilon(fairness, buckets_amount)

    return epsilon, query_time / len(points)
