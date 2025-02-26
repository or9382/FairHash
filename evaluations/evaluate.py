from utils import get_bucket_group_matrix, calculate_pairwise_fairness_list, calculate_epsilon


def evaluate(cdf_based_hash, ranking_based_hash, cut_based_hash, points, buckets_amount, groups):
    cdf_buckets = {i: [] for i in range(5)}
    cut_buckets = {i: [] for i in range(5)}
    ranking_buckets = {i: [] for i in range(5)}

    for p in points:
        cdf_bucket = cdf_based_hash(p)
        cut_bucket = cut_based_hash(p)
        ranking_bucket = ranking_based_hash(p)

        cdf_buckets[cdf_bucket].append(p)
        cut_buckets[cut_bucket].append(p)
        ranking_buckets[ranking_bucket].append(p)

    cdf_a = get_bucket_group_matrix(list(list(bucket) for bucket in cdf_buckets.values()), groups)
    cut_a = get_bucket_group_matrix(list(list(bucket) for bucket in cut_buckets.values()), groups)
    rank_a = get_bucket_group_matrix(list(list(bucket) for bucket in ranking_buckets.values()), groups)

    cdf_fairness = calculate_pairwise_fairness_list(cdf_a, groups)
    cut_fairness = calculate_pairwise_fairness_list(cut_a, groups)
    rank_fairness = calculate_pairwise_fairness_list(rank_a, groups)

    cdf_epsilon = calculate_epsilon(cdf_fairness, 5)
    cut_epsilon = calculate_epsilon(cut_fairness, 5)
    rank_epsilon = calculate_epsilon(rank_fairness, 5)
