import itertools
import os

import numpy as np
import pandas as pd

from algorithms import create_cdf_hash_function, create_ranking_hash_function, create_cut_hash_function
from evaluations.evaluation import measure_time, test

csv_path = os.path.join(os.path.dirname(__file__), r'dataset\compas-scores-raw.csv')


def prepare_data():
    # Load dataset from a CSV file
    compas = pd.read_csv(csv_path, index_col=False).dropna()

    # Extract Person_ID and AssessmentID columns
    person_ids = np.array(compas['Person_ID'])
    assessment_ids = np.array(compas['AssessmentID'])

    # Normalize person_ids and assessment_ids to [0, 1] using Min-Max scaling
    compas['person_ids'] = (person_ids - person_ids.min()) / (person_ids.max() - person_ids.min())
    compas['assessment_ids'] = (assessment_ids - assessment_ids.min()) / (assessment_ids.max() - assessment_ids.min())

    compas = compas.drop_duplicates(subset=['person_ids', 'assessment_ids']).sample(frac=1, random_state=42).reset_index(drop=True)

    groups = []
    for sex, race in itertools.product(
            ['Male', 'Female'],
            ['African-American', 'Caucasian', 'Hispanic']
    ):
        group = compas.loc[(compas['Sex_Code_Text'] == sex) & (compas['Ethnic_Code_Text'] == race)]
        groups.append(group[['assessment_ids', 'person_ids']].values.tolist())

    return groups


def compas_test():
    bucket_amount = 5

    black_males, white_males, hispanic_males, black_females, white_females, hispanic_females = prepare_data()
    # We use (about) 20% of all data and keep a minority to majority ratio of 0.25
    black_males = black_males[:200]
    white_males = white_males[:200]
    hispanic_males = hispanic_males[:150]
    black_females = black_females[:150]
    white_females = white_females[:150]
    hispanic_females = hispanic_females[:50]

    black_males = [(m, 0) for m in black_males]
    white_males = [(m, 1) for m in white_males]
    hispanic_males = [(m, 2) for m in hispanic_males]
    black_females = [(f, 3) for f in black_females]
    white_females = [(f, 4) for f in white_females]
    hispanic_females = [(f, 5) for f in hispanic_females]

    combined = black_males + white_males + hispanic_males + black_females + white_females + hispanic_females
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
        [len(black_males), len(white_males), len(hispanic_males), len(black_females), len(white_females), len(hispanic_females)]
    )

    black_male_sample = black_males[:20]
    white_male_sample = white_males[:20]
    hispanic_male_sample = hispanic_males[:15]
    black_female_sample = black_females[:15]
    white_female_sample = white_females[:15]
    hispanic_female_sample = hispanic_females[:5]

    combined_sample = black_male_sample + white_male_sample + hispanic_male_sample + black_female_sample + white_female_sample + hispanic_female_sample
    np.random.shuffle(combined_sample)

    rank_100_hash, rank_100_preprocessing = measure_time(
        create_ranking_hash_function,
        combined_sample,
        bucket_amount,
        [len(black_male_sample), len(white_male_sample), len(hispanic_male_sample), len(black_female_sample), len(white_female_sample), len(hispanic_female_sample)]
    )

    cdf_epsilon, cdf_query_time = test(cdf_hash, combined, bucket_amount, [len(black_males), len(white_males)])
    cut_epsilon, cut_query_time = test(cut_hash, combined, bucket_amount, [len(black_males), len(white_males)])
    rank_100_epsilon, rank_100_query_time = test(rank_100_hash, combined, bucket_amount, [len(black_males), len(white_males)])

    print(f"CDF: {cdf_epsilon=}; {cdf_preprocessing=}; {cdf_query_time=}")
    print(f"cut: {cut_epsilon=}; {cut_preprocessing=}; {cut_query_time=}")
    print(f"rank 100: {rank_100_epsilon=}; {rank_100_preprocessing=}; {rank_100_query_time=}")


if __name__ == '__main__':
    compas_test()
