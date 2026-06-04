import json
from sklearn.preprocessing import RobustScaler
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np

### 1) collect metrics from outputs 
# - random seeds
# - parametre configurations
# - output metrics 
# - trial block
# segment first 300 as Markov
# segment second 300 as State
# segment third 300 as General
# segment last 600 as All

global_init_params_path = "..\..\corpus\pilot_study_2\\trial_data\params\\20260604_195844_trial.json"
with open(global_init_params_path, "r") as f:
    global_init_params_data = json.load(f)

trial_logs_path = "..\..\corpus\pilot_study_2\\trial_data\logs\\20260604_195844_trial.jsonl"

# See trials for details on trial blocks
all_trials = []
markov_trials = []
state_trials = []
gs_trials = []
all_rand_trials = []

with open(trial_logs_path, "r") as f:
    for idx, line in enumerate(f):
        cleaned_line = line.strip()
        if cleaned_line:
            dic = json.loads(cleaned_line)
            dic["index"] = idx
            if idx < 300: 
                markov_trials.append(dic)
            elif 300 <= idx < 600:
                state_trials.append(dic)
            elif 600 <= idx < 900:
                gs_trials.append(dic)
            else:
                all_rand_trials.append(dic)
            all_trials.append(dic)

list_metrics = []
for trial in all_trials:
    metrics = trial["metrics"]
    list_metrics.append({
        "index": trial["index"],
        "centroid_mean": metrics["centroid_mean"], 
        "centroid_std": metrics["centroid_std"], 
        "centroid_skewness": metrics["centroid_skewness"], 
        "flatness_mean": metrics["flatness_mean"], 
        "flatness_std": metrics["flatness_std"], 
        "flatness_skewness": metrics["flatness_skewness"], 
        "flux_mean": metrics["flux_mean"], 
        "flux_std": metrics["flux_std"], 
        "flux_skewness": metrics["flux_skewness"]
    })

metrics_df_all = pd.DataFrame(list_metrics)
print(metrics_df_all.head())
scaler = RobustScaler()
metrics_to_scale = metrics_df_all.iloc[:,1 :]
scaled_metrics = scaler.fit_transform(metrics_to_scale)

markov_scaled_metrics = scaled_metrics[:300]
markov_cosine_matrix = []
for i in range(markov_scaled_metrics.shape[-1]):
    row = []
    for j in range(markov_scaled_metrics.shape[-1]):
        cosine_dist = cosine(markov_scaled_metrics[i], markov_scaled_metrics[j])
        row.append(cosine_dist)
    markov_cosine_matrix.append(row)
markov_cosine_matrix = np.array(markov_cosine_matrix)

flat_markov_cosine_distances = markov_cosine_matrix[np.triu_indices(markov_cosine_matrix.shape[0], k = 1)]

expected_length = int((markov_scaled_metrics.shape[-1] * (markov_scaled_metrics.shape[-1] - 1)) / 2)
actual_length = len(flat_markov_cosine_distances)

assert actual_length == expected_length, "Data shape mismatch!"

mean_diversity = np.mean(flat_markov_cosine_distances)
std_diversity  = np.std(flat_markov_cosine_distances)
max_diversity  = np.max(flat_markov_cosine_distances)
print(
    mean_diversity,
    std_diversity,
    max_diversity
)
# Normalize the metrics

### 2) com