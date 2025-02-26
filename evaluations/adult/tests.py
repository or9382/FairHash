import os

import numpy as np
import pandas as pd

from algorithms import create_cdf_hash_function, create_ranking_hash_function, create_cut_hash_function


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

    adult = adult.drop_duplicates(subset=['age', 'fnlwgt'])

    males = adult.loc[adult['sex'] == 'Male'][['fnlwgt', 'age']].values.tolist()
    females = adult.loc[adult['sex'] == 'Female'][['fnlwgt', 'age']].values.tolist()

    return males, females


def train():
    males, females = prepare_data()
    # We use (about) 20% of all data and keep a minority to majority ratio of 0.25
    males, females = males[:4400], females[:1100]
    males, females = [(m, 0) for m in males], [(f, 1) for f in females]

    cdf_hash = create_cdf_hash_function(males + females, 100)
    cut_hash = create_cut_hash_function(males + females, 100, [len(males), len(females)])

    male_sample, female_sample = males[:80], females[:20]
    rank_100_hash = create_ranking_hash_function(
        male_sample + female_sample,
        100,
        [len(male_sample), len(female_sample)]
    )

    male_sample, female_sample = males[:800], females[:200]
    rank_1000_hash = create_ranking_hash_function(
        male_sample + female_sample,
        100,
        [len(male_sample), len(female_sample)]
    )

    pass


if __name__ == '__main__':
    train()
