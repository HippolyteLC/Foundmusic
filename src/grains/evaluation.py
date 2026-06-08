import json
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, Normalizer
import pandas as pd
from scipy.spatial.distance import cosine
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap


### IMPORTANT: every unqiue config's K repetitions are aggregated and the mean is taken
# This is done to keep the iid property for statistical testing of variety of outputs

STUDY_NAME = "pilot_study_3"
TRIAL_NAME = "20260605_112241_trial"
GLOBAL_INIT_PARAMS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\params\\{TRIAL_NAME}.json"
with open(GLOBAL_INIT_PARAMS_PATH, "r") as f:
    global_init_params_data = json.load(f)

TRIAL_LOGS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\logs\\{TRIAL_NAME}.jsonl"

RESULTS_DIR = f"..\..\corpus\{STUDY_NAME}\\trial_data\\results\\"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

FIGURES_DIR = f"..\..\corpus\{STUDY_NAME}\\trial_data\\figures\\"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

TRIALS_PARAMS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\params\\{TRIAL_NAME}.json"
with open(TRIALS_PARAMS_PATH, "r") as f:
    K_REP = json.load(f)["general_trial_parametres"]["k_repetitions"]

# See trials for details on trial blocks
all_trials = []
markov_trials = []
state_trials = []
gs_trials = []
all_rand_trials = []

metrics_aggregate_per_config = np.zeros(3)

with open(TRIAL_LOGS_PATH, "r") as f:
    
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


def aggregate_metrics_per_config(arr, k_rep=K_REP):
    trial_aggr = []
    metrics_keys = list(arr[0]["metrics"].keys())
    len_metrics = len(metrics_keys)
    aggr_metrics = np.zeros(len_metrics)
    trial_ids = []
    for idx, trial in enumerate(arr):
        config_id = int(idx//k_rep) 
        metrics_values = np.array(list(trial["metrics"].values()))
        aggr_metrics = aggr_metrics + metrics_values
        trial_ids.append(trial["trial_id"])
        if ((idx + 1) % k_rep) == 0:
            mean_metrics = aggr_metrics / k_rep
            trial_aggr.append(
                {
                    "trial_ids": trial_ids,
                    "config_id": config_id,
                    "metrics": {metrics_keys[i]: mean_metrics[i] for i in range(len_metrics)}
                }
            )
            aggr_metrics = np.zeros(len_metrics)
            trial_ids = []
    return trial_aggr

### Collect trial metrics data

def trial_to_scaled_metric_df(list_trials, scaler=1):
    """
    Robust scaler as default for outlier values. 
    Computes scaled metrics df per trial list. 
    """
    aggr_metric = aggregate_metrics_per_config(list_trials)
    scalers = [StandardScaler, RobustScaler, PowerTransformer, Normalizer]
    scaler_ = scalers[scaler]()
    list_metrics = []
    for config in aggr_metric:
        metrics = config["metrics"]
        list_metrics.append({
            # "config_id": config["config_id"],
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

metrics_df_all, scaled_metrics_df_all, scaled_metrics_arr_all = trial_to_scaled_metric_df(all_trials)
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

### analyze data
# TODO: produce histograms to display the 9 output metrics per parametre subgroup. 

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
        
        print("Interpretation:     SIGNIFICANT. The choice of parameter subgroup produces ")
        print("                    statistically significant differences in this acoustic ")
        print("                    property across configurations.")
    else:
        print("Interpretation:     NOT SIGNIFICANT. The random sampling seeds cause more ")
        print("                    variance than the actual parameter adjustments.")
        
    return {"f_stat": f_statistic, "p_val": p_value, "df_between": df_between, "df_within": df_within}

all_metrics_dfs = [metrics_df_markov, metrics_df_state, metrics_df_gs]
all_scaled_metrics_dfs = [scaled_metrics_df_markov, scaled_metrics_df_state, scaled_metrics_df_gs]
metrics_labels = metrics_df_markov.columns

anovas_per_metric = {}
for col in metrics_labels:
    anovas_per_metric[col] = output_anova_results(all_scaled_metrics_dfs, col) #{"f_statistic": f_statistic, "p_value": p_value}

results_data = {
    "levine_results": levene_output,
    "anova_results": anovas_per_metric,
}
RESULTS_FILE_NAME = f"{TRIAL_NAME}.json"
with open(os.path.normpath(RESULTS_DIR + RESULTS_FILE_NAME), "w") as f:
    json.dump(results_data, f, indent=4)

### Creating + saving box_plots

data_to_plot = [
    markov_flattened_matrix,
    state_flattened_matrix,
    gs_flattened_matrix,
    all_rand_flattened_matrix
]

PLOT_FILE_NAME = f"{TRIAL_NAME}_n_box_{len(data_to_plot)}_box_plot.png"
labels = ['Markov \nGroup', 'State-Dependent \nGroup', 'Granular Synthesis \nGroup', 'Baseline Group \n(Fully Randomized)']

plt.figure(figsize=(8, 6))

# Create the boxplot
plt.boxplot(data_to_plot, labels=labels)
plt.ylabel('Pairwise Cosine Distance', fontsize=12)
plt.title('Acoustic Diversity Profile across Parameter Subgroups', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.normpath(FIGURES_DIR + PLOT_FILE_NAME), dpi=300)
# plt.show()

### Creating + saving reduced diminensionality scatter plot of outputs.

print("starting umap process")

PLOT_FILE_NAME_PNG = f"{TRIAL_NAME}_umap_acoustic_metrics_space.png"
PLOT_FILE_NAME_PDF = f"{TRIAL_NAME}_umap_acoustic_metrics_space.pdf"

metrics = list(metrics_df_markov.columns)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(scaled_metrics_arr_all)

len_markov  = len(metrics_df_markov)
len_state   = len(metrics_df_state)
len_general = len(metrics_df_gs)
len_random  = len(metrics_df_all_rand)

group_labels = np.repeat(
    ['Markov Group', 'State-Dependent Group', 'Granular Synthesis Group', 'Baseline Group'],
    [len_markov, len_state, len_general, len_random]
)

metrics_df_all['parameter_group'] = group_labels

metrics_df_all['umap_x'] = embedding[:, 0]
metrics_df_all['umap_y'] = embedding[:, 1]

custom_colors = {
    'Markov Group': "#0e5b92",           # Deep Blue (Structured)
    'State-Dependent Group': "#ff420e",        # Bright Orange (Structured)
    'Granular Synthesis Group': '#2ca02c',       # Forest Green (Structured)
    'Baseline Group': "#E3D927" # Muted Slate Gray (Your background control chaos)
}

print("output umap")

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='umap_x', y='umap_y', 
    hue='parameter_group', 
    alpha=0.6, 
    palette=custom_colors, 
    data=metrics_df_all
)

plt.title('UMAP Projection of Generative Granular Acoustic Space', fontsize=14, fontweight='bold')
plt.xlabel('UMAP Axis 1')
plt.ylabel('UMAP Axis 2')
plt.legend(title='Parameter Subgroup')
plt.grid(True, alpha=0.3)

plt.savefig(os.path.normpath(FIGURES_DIR + PLOT_FILE_NAME_PDF), format='pdf', bbox_inches='tight')

plt.savefig(os.path.normpath(FIGURES_DIR + PLOT_FILE_NAME_PNG), format='png', dpi=300, bbox_inches='tight')


