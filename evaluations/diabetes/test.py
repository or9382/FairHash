import os

import numpy as np
import pandas as pd

from algorithms import create_cdf_hash_function, create_ranking_hash_function, create_cut_hash_function
from evaluations.evaluation import measure_time, test

csv_path = os.path.join(os.path.dirname(__file__), r'dataset\diabetic_data.csv')


def prepare_data():
    # Load dataset from a CSV file
    diabetes = pd.read_csv(csv_path, index_col=False)[['encounter_id', 'patient_nbr', 'gender']].dropna()

    # Extract Person_ID and AssessmentID columns
    encounter_ids = np.array(diabetes['encounter_id'])
    patient_nbrs = np.array(diabetes['patient_nbr'])

    # Normalize encounter_ids and patient_nbrs to [0, 1] using Min-Max scaling
    diabetes['encounter_ids'] = (encounter_ids - encounter_ids.min()) / (encounter_ids.max() - encounter_ids.min())
    diabetes['patient_nbrs'] = (patient_nbrs - patient_nbrs.min()) / (patient_nbrs.max() - patient_nbrs.min())

    males = diabetes.loc[diabetes['gender'] == 'Male'][['encounter_ids', 'patient_nbrs']].values.tolist()
    females = diabetes.loc[diabetes['gender'] == 'Female'][['encounter_ids', 'patient_nbrs']].values.tolist()

    return males, females


def diabetes_test():
    bucket_amount = 5

    males, females = prepare_data()
    # We use (about) 20% of all data and keep a minority to majority ratio of 0.25
    males, females = males[:1600], females[:4000]
    males, females = [(m, 0) for m in males], [(f, 1) for f in females]
    combined = males + females
    np.random.shuffle(combined)

    cdf_hash, cdf_preprocessing = measure_time(
        create_cdf_hash_function,
        combined,
        bucket_amount
    )
    cut_hash, cut_preprocessing = measure_time(
        create_cut_hash_function,
        combined,
        bucket_amount,
        [len(males), len(females)]
    )

    male_sample, female_sample = males[:80], females[:20]
    combined_sample = male_sample + female_sample
    np.random.shuffle(combined_sample)

    rank_100_hash, rank_100_preprocessing = measure_time(
        create_ranking_hash_function,
        combined_sample,
        bucket_amount,
        [len(male_sample), len(female_sample)]
    )

    cdf_epsilon, cdf_query_time = test(cdf_hash, combined, bucket_amount, [len(males), len(females)])
    cut_epsilon, cut_query_time = test(cut_hash, combined, bucket_amount, [len(males), len(females)])
    rank_100_epsilon, rank_100_query_time = test(rank_100_hash, combined, bucket_amount, [len(males), len(females)])

    print(f"CDF: {cdf_epsilon=}; {cdf_preprocessing=}; {cdf_query_time=}")
    print(f"cut: {cut_epsilon=}; {cut_preprocessing=}; {cut_query_time=}")
    print(f"rank 100: {rank_100_epsilon=}; {rank_100_preprocessing=}; {rank_100_query_time=}")


if __name__ == '__main__':
    diabetes_test()
