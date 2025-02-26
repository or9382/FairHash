from algorithms.cdf import create_hash_function as create_cdf_hash_function
from algorithms.ranking import create_hash_function as create_ranking_hash_function
from algorithms.cut import create_hash_function as create_cut_hash_function
from utils import get_bucket_group_matrix, calculate_pairwise_fairness_list, calculate_epsilon


def main():
    # Split data into groups based on the 'sex' column
    groups = [
        adult.loc[adult['sex'] == sex, ['fnlwgt', 'age']].values.tolist()[:500]
        for sex in adult['sex'].unique()
    ]

    # Flatten the groups to get all points
    points = [point for group in groups for point in group]

    cdf_based_hash = create_cdf_hash_function(points, 5)
    cut_based_hash = create_cut_hash_function(points, 5, groups)
    rank_based_hash = create_ranking_hash_function(points, 5, groups)

    cdf_buckets = {i: [] for i in range(5)}
    cut_buckets = {i: [] for i in range(5)}
    rank_buckets = {i: [] for i in range(5)}

    for p in points:
        cdf_bucket = cdf_based_hash(p)
        cut_bucket = cut_based_hash(p)
        rank_bucket = rank_based_hash(p)

        cdf_buckets[cdf_bucket].append(p)
        cut_buckets[cut_bucket].append(p)
        rank_buckets[rank_bucket].append(p)

        # print(f"{cdf_bucket = }")
        # print(f"{cut_bucket = }")
        # print(f"{rank_bucket = }")

    cdf_a = get_bucket_group_matrix(list(list(bucket) for bucket in cdf_buckets.values()), groups)
    cut_a = get_bucket_group_matrix(list(list(bucket) for bucket in cut_buckets.values()), groups)
    rank_a = get_bucket_group_matrix(list(list(bucket) for bucket in rank_buckets.values()), groups)

    cdf_fairness = calculate_pairwise_fairness_list(cdf_a, groups)
    cut_fairness = calculate_pairwise_fairness_list(cut_a, groups)
    rank_fairness = calculate_pairwise_fairness_list(rank_a, groups)

    cdf_epsilon = calculate_epsilon(cdf_fairness, 5)
    cut_epsilon = calculate_epsilon(cut_fairness, 5)
    rank_epsilon = calculate_epsilon(rank_fairness, 5)

    pass


if __name__ == '__main__':
    main()
