import json
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, Normalizer
import pandas as pd
from scipy.spatial.distance import cosine
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

### 1) collect metrics from outputs 
# - random seeds
# - parametre configurations
# - output metrics 
# - trial block
# segment first 300 as Markov
# segment second 300 as State
# segment third 300 as General
# segment last 600 as All

trial_name = "20260604_195844_trial"
global_init_params_path = f"..\..\corpus\pilot_study_2\\trial_data\params\\{trial_name}.json"
with open(global_init_params_path, "r") as f:
    global_init_params_data = json.load(f)

trial_logs_path = f"..\..\corpus\pilot_study_2\\trial_data\logs\\{trial_name}.jsonl"

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

### Collect trial metrics data

def trial_to_scaled_metric_df(list_trials, scaler=1):
    """
    Robust scaler as default for outlier values. 
    Computes scaled metrics df per trial list. 
    """
    scalers = [StandardScaler, RobustScaler, PowerTransformer, Normalizer]
    scaler_ = scalers[scaler]()
    list_metrics = []
    for trial in list_trials:
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
    metrics_df = pd.DataFrame(list_metrics)
    metrics_to_scale = metrics_df.iloc[:,1 :]
    scaled_metrics = scaler_.fit_transform(metrics_to_scale)
    scaled_metrics_df = pd.DataFrame(scaled_metrics, columns=metrics_to_scale.columns)
    return metrics_to_scale, scaled_metrics_df, scaled_metrics

def compute_cosine_matrix(df):
    cosine_matrix = []
    for i in range(df.shape[-1]):
        row = []
        for j in range(df.shape[-1]):
            cosine_dist = cosine(df[i], df[j])
            row.append(cosine_dist)
        cosine_matrix.append(row)
    cosine_matrix = np.array(cosine_matrix)
    return cosine_matrix

def flatten_upper_half(matrix):
    flat_cosine_distances = matrix[np.triu_indices(matrix.shape[0], k = 1)]
    expected_length = int((matrix.shape[-1] * (matrix.shape[-1] - 1)) / 2)
    actual_length = len(flat_cosine_distances)
    assert actual_length == expected_length, "Data shape mismatch!"
    return flat_cosine_distances

def compute_stats(arr):
    mean_diversity = np.mean(arr)
    std_diversity  = np.std(arr)
    max_diversity  = np.max(arr)
    return mean_diversity, std_diversity, max_diversity

metrics_df_markov, scaled_metrics_df_markov, scaled_metrics_arr_markov = trial_to_scaled_metric_df(markov_trials)
metrics_df_state, scaled_metrics_df_state, scaled_metrics_arr_state = trial_to_scaled_metric_df(state_trials)
metrics_df_gs, scaled_metrics_df_gs, scaled_metrics_arr_gs = trial_to_scaled_metric_df(gs_trials)
metrics_df_all_rand, scaled_metrics_df_all_rand, scaled_metrics_arr_all_rand = trial_to_scaled_metric_df(all_rand_trials)

markov_flattened_matrix = flatten_upper_half(compute_cosine_matrix(scaled_metrics_arr_markov))
markov_mean_diversity, markov_std_diversity, markov_max_diversity = compute_stats(markov_flattened_matrix)

state_flattened_matrix = flatten_upper_half(compute_cosine_matrix(scaled_metrics_arr_state))
state_mean_diversity, state_std_diversity, state_max_diversity = compute_stats(state_flattened_matrix)

gs_flattened_matrix = flatten_upper_half(compute_cosine_matrix(scaled_metrics_arr_gs))
gs_mean_diversity, gs_std_diversity, gs_max_diversity = compute_stats(gs_flattened_matrix)

all_rand_flattened_matrix = flatten_upper_half(compute_cosine_matrix(scaled_metrics_arr_all_rand))
all_rand_mean_diversity, all_rand_std_diversity, all_rand_max_diversity = compute_stats(all_rand_flattened_matrix)

### Get results from cosine_dist metrics (ANOVA + Levine)


levene_stat, levene_p = stats.levene(markov_flattened_matrix, state_flattened_matrix, gs_flattened_matrix)
def format_p_value(p):
    return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
print("--- Statistical Significance Tests ---")
print(f"Levene's Test (Variance Significance): Stat={levene_stat:.4f}, p-value={format_p_value(levene_p)}")

levene_output = {"levene_stat": levene_stat, "levene_p": levene_p}

if levene_p < 0.05:
    print("Result: Success! The difference in output diversity between your groups is highly significant.")

def compute_one_way_anova(dfs, metric):
    f_statistic, p_value = stats.f_oneway(dfs[0][metric], dfs[1][metric], dfs[2][metric])
    return f_statistic, p_value

def output_anova_results(all_metrics_dfs, metric):
    f_statistic, p_value = compute_one_way_anova(all_metrics_dfs, metric)
    num_dfs = len(all_metrics_dfs)
    total_samples = len(all_metrics_dfs[0]) + len(all_metrics_dfs[1]) + len(all_metrics_dfs[2])
    df_between = num_dfs - 1
    df_within = total_samples - num_dfs
    print(f"=== One-Way ANOVA Results for Axis: {metric} ===")
    print(f"Degrees of Freedom:  F({df_between}, {df_within})")
    print(f"F-Statistic:         {f_statistic:.4f}")
    print(f"p-value:             {format_p_value(p_value)}")
    if p_value < 0.05:
        print("Interpretation:     SIGNIFICANT. The choice of parameter subgroup exerts a ")
        print("                    verifiable structural control over this acoustic axis that ")
        print("                    far outweighs the random seed noise of the system.")
    else:
        print("Interpretation:     NOT SIGNIFICANT. The random sampling seeds cause more ")
        print("                    variance than the actual parameter adjustments.")
        
    return {"f_stat": f_statistic, "p_val": p_value, "df_between": df_between, "df_within": df_within}

all_metrics_dfs = [metrics_df_markov, metrics_df_state, metrics_df_gs]
all_scaled_metrics_dfs = [scaled_metrics_df_markov, scaled_metrics_df_state, scaled_metrics_df_gs]

anovas_per_metric = {}
for col in metrics_df_markov.columns:
    anovas_per_metric[col] = output_anova_results(all_scaled_metrics_dfs, col) #{"f_statistic": f_statistic, "p_value": p_value}

RESULTS_DIR = "..\..\corpus\pilot_study_2\\trial_data\\results\\"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
RESULTS_FILE_NAME = f"{trial_name}.json"


results_data = {
    "levine_results": levene_output,
    "anova_results": anovas_per_metric,
}
with open(os.path.normpath(RESULTS_DIR + RESULTS_FILE_NAME), "w") as f:
    json.dump(results_data, f, indent=4)

### Creating + saving box_plots

FIGURES_DIR = "..\..\corpus\pilot_study_2\\trial_data\\figures\\"
PLOT_FILE_NAME = f"{trial_name}.png"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

data_to_plot = [
    markov_flattened_matrix,
    state_flattened_matrix,
    gs_flattened_matrix
]

plt.figure(figsize=(8, 6))

# Create the boxplot
plt.boxplot(data_to_plot, labels=['Markov Group', 'State-Dependent Group', 'General Granular Synthesis Group'])
plt.ylabel('Pairwise Cosine Distance', fontsize=12)
plt.title('Acoustic Diversity Profile across Parameter Subgroups', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.normpath(FIGURES_DIR + PLOT_FILE_NAME), dpi=300)
plt.show()

