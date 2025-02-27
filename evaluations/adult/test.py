import os

import numpy as np
import pandas as pd

from algorithms import create_cdf_hash_function, create_ranking_hash_function, create_cut_hash_function
from evaluations.evaluation import measure_time, test

csv_path = os.path.join(os.path.dirname(__file__), r'dataset\adult.data')


def prepare_data():
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]

    # Load dataset from a CSV file
    adult = pd.read_csv(csv_path, names=names, index_col=False, skipinitialspace=True).dropna()

    # Extract age and fnlwgt columns
    age = np.array(adult['age'])
    fnlwgt = np.array(adult['fnlwgt'])

    # Normalize age and fnlwgt to [0, 1] using Min-Max scaling
    adult['age'] = (age - age.min()) / (age.max() - age.min())
    adult['fnlwgt'] = (fnlwgt - fnlwgt.min()) / (fnlwgt.max() - fnlwgt.min())

    adult = adult.drop_duplicates(subset=['fnlwgt']).sample(frac=1, random_state=42).reset_index(drop=True)

    males = adult.loc[adult['sex'] == 'Male'][['fnlwgt', 'age']].values.tolist()
    females = adult.loc[adult['sex'] == 'Female'][['fnlwgt', 'age']].values.tolist()

    return males, females


def train():
    bucket_amount = 5

    males, females = prepare_data()
    # We use (about) 20% of all data and keep a minority to majority ratio of 0.25
    males, females = males[:4400], females[:1100]
    males, females = [(m, 0) for m in males], [(f, 1) for f in females]
    combined = males + females
    np.random.shuffle(combined)

    cdf_hash, cdf_preprocesing = measure_time(
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

    print(f"CDF: {cdf_epsilon=}; {cdf_preprocesing=}; {cdf_query_time=}")
    print(f"cut: {cut_epsilon=}; {cut_preprocessing=}; {cut_query_time=}")
    print(f"rank 100: {rank_100_epsilon=}; {rank_100_preprocessing=}; {rank_100_query_time=}")


if __name__ == '__main__':
    train()
