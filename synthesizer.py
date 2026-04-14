import numpy as np
import audioflux as af
import pandas as pd
import json
from scipy import stats


def load_json_to_df(fp):
    with open(fp) as f:
        df = pd.read_json(f)
    return df


def get_distribution(df, d1,d2,metadata):
    df = load_json_to_df(metadata)
    df = df.fillna(0)
    cols_to_check = df.iloc[:, 3:]
    df = df[cols_to_check.ne(0).any(axis=1)]
    feature_data = df[[d1, d2]].values.T
    kde = stats.gaussian_kde(feature_data)
    weights = kde.evaluate(feature_data)
    probs = weights / weights.sum()
    return df, probs

seed = np.random.seed(1)

def distr_concat(input_path, output_path, metadata, output_length, d1, d2):
    y,sr = af.read(input_path)
    if y.ndim >1:
        y = np.mean(y, axis=1)
    if not output_length:
        output_length = len(y)
    output_buffer = np.array([])

    df = load_json_to_df(metadata)
    df, probs = get_distribution(df,d1, d2, metadata) 
    while len(output_buffer) < output_length:
        sampled_grain = df.sample(n=1, weights=probs, random_state=seed)
        # print(sampled_grain)
        s, e = int(sampled_grain["grain_start"]), int(sampled_grain["grain_start"]+sampled_grain["grain_size"])
        grain = y[s: e]
        grain = np.hanning(len(grain)) * np.array(grain)
        # print(grain, output_buffer)
        output_buffer = np.concatenate([output_buffer, grain]) 
    af.write(output_path, output_buffer, samplerate=sr)

 
 
