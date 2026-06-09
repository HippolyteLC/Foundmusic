import json
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, Normalizer
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import cosine
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors  
import seaborn as sns
import os
import umap
from analysis import get_histograms, get_scatter_plt
from scikit_posthocs import posthoc_dunn

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### IMPORTANT: every unqiue config's K repetitions are aggregated and the mean is taken
# This is done to keep the iid property for statistical testing of variety of outputs

STUDY_NAME = "pilot_study_1"
TRIAL_NAME = "20260608_130222_trial"
GLOBAL_INIT_PARAMS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\params\\{TRIAL_NAME}.json"
with open(GLOBAL_INIT_PARAMS_PATH, "r") as f:
    global_init_params_data = json.load(f)

TRIAL_LOGS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\logs\\{TRIAL_NAME}.jsonl"

RESULTS_DIR = f"..\..\corpus\{STUDY_NAME}\\trial_data\\results\\"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

FIGURES_DIR = f"..\..\corpus\{STUDY_NAME}\\trial_data\\figures\\{TRIAL_NAME}\\"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

TRIALS_PARAMS_PATH = f"..\..\corpus\{STUDY_NAME}\\trial_data\params\\{TRIAL_NAME}.json"
with open(TRIALS_PARAMS_PATH, "r") as f:
    data_dict = json.load(f)
    K_REP = data_dict["general_trial_parametres"]["k_repetitions"]
    # N_CONFIGS_PER_PARAM_GROUP = data_dict["general_trial_parametres"][""]
    # TODO: added dynamic loading of configs per group

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
            all_trials.append(dic)

def aggregate_metrics_per_config(arr, k_rep=K_REP):
    trial_aggr = []
    metrics_keys = list(arr[0]["metrics"].keys())
    len_metrics = len(metrics_keys)
    
    config_metrics_list = []
    trial_ids = []
    
    for idx, trial in enumerate(arr):
        config_id = int(idx//k_rep) 
        metrics_values = np.array(list(trial["metrics"].values()))
        
        config_metrics_list.append(metrics_values)
        trial_ids.append(trial["trial_id"])
        
        if ((idx + 1) % k_rep) == 0:
            mean_metrics = np.mean(config_metrics_list, axis=0)
            # std_metrics = np.std(config_metrics_list, axis=0)
            
            combined_metrics = {}
            for i in range(len_metrics):
                combined_metrics[f"{metrics_keys[i]}"] = mean_metrics[i]
                # combined_metrics[f"{metrics_keys[i]}_std"] = std_metrics[i]
            
            trial_aggr.append(
                {
                    "trial_ids": trial_ids,
                    "config_id": config_id,
                    "metrics": combined_metrics
                }
            )
            
            config_metrics_list = []
            trial_ids = []
            
    return trial_aggr

all_trials_aggregated = aggregate_metrics_per_config(all_trials)
# print(all_trials_aggregated[0])

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Collect trial metrics data

def trial_to_scaled_metric_df(list_trials, scaler=1):
    """
    Robust scaler as default for outlier values. 
    Computes scaled metrics df per trial list. 
    """
    list_trials
    scalers = [StandardScaler, RobustScaler, PowerTransformer, Normalizer]
    scaler_ = scalers[scaler]()
    list_metrics = []
    for config in list_trials:
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
    # metrics_to_scale = metrics_df.iloc[:,1 :]
    scaled_metrics = scaler_.fit_transform(metrics_df)
    scaled_metrics_df = pd.DataFrame(scaled_metrics, columns=metrics_df.columns)
    return metrics_df, scaled_metrics_df, scaled_metrics

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


df_trials_aggregated, df_scaled_trials_aggregated, arr_scaled_trials_aggregated = trial_to_scaled_metric_df(all_trials_aggregated)
print(arr_scaled_trials_aggregated.shape)

markov_flattened_matrix = flatten_upper_half(compute_cosine_matrix(arr_scaled_trials_aggregated[:200]))
markov_mean_diversity, markov_std_diversity, markov_max_diversity = compute_stats(markov_flattened_matrix)

state_flattened_matrix = flatten_upper_half(compute_cosine_matrix(arr_scaled_trials_aggregated[200:400]))
state_mean_diversity, state_std_diversity, state_max_diversity = compute_stats(state_flattened_matrix)

gs_flattened_matrix = flatten_upper_half(compute_cosine_matrix(arr_scaled_trials_aggregated[400:600]))
gs_mean_diversity, gs_std_diversity, gs_max_diversity = compute_stats(gs_flattened_matrix)

all_rand_flattened_matrix = flatten_upper_half(compute_cosine_matrix(arr_scaled_trials_aggregated[600:]))
all_rand_mean_diversity, all_rand_std_diversity, all_rand_max_diversity = compute_stats(all_rand_flattened_matrix)

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### analyze data
# TODO: produce histograms to display the 9 output metrics per parametre subgroup. 

metrics_labels = list(df_scaled_trials_aggregated.columns)

get_histograms(FIGURES_DIR, "markov_outputs_histograms_.png",df_scaled_trials_aggregated[:200], metrics_labels)
get_histograms(FIGURES_DIR, "state_outputs_histograms_.png", df_scaled_trials_aggregated[200:400], metrics_labels)
get_histograms(FIGURES_DIR, "gs_outputs_histograms_.png", df_scaled_trials_aggregated[400:600], metrics_labels)
get_histograms(FIGURES_DIR, "all_random_outputs_histograms_.png", df_scaled_trials_aggregated[:600], metrics_labels)

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Get results from cosine_dist metrics (ANOVA + Levine)

def format_p_value(p):
    return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"


### ___ LEVENE'S ___
# levene_stat, levene_p = stats.levene(markov_flattened_matrix, state_flattened_matrix, gs_flattened_matrix)

# print("--- Statistical Significance Tests ---")
# print(f"Levene's Test (Variance Significance): Stat={levene_stat:.4f}, p-value={format_p_value(levene_p)}")

# levene_output = {"levene_stat": levene_stat, "levene_p": levene_p}

# if levene_p < 0.05:
#     print("Result: Success! The difference in output diversity between your groups is highly significant.")

def compute_kruskal_wallis_one_way(dfs, metric):
    H_statistic, p_value = stats.kruskal(dfs[0][metric], dfs[1][metric], dfs[2][metric])
    return H_statistic, p_value

def output_kruskal_wallis_results(dfs, metric):
    H_statistic, p_value = compute_kruskal_wallis_one_way(dfs, metric)
    return {"H_stat": H_statistic, "p_val": p_value}

def compute_one_way_anova(dfs, metric):
    f_statistic, p_value = stats.f_oneway(dfs[0][metric], dfs[1][metric], dfs[2][metric])
    return f_statistic, p_value

def output_anova_results(all_metrics_dfs, metric):
    f_statistic, p_value = compute_one_way_anova(all_metrics_dfs, metric)
    num_dfs = len(all_metrics_dfs)
    return {"f_stat": f_statistic, "p_val": p_value}

def compute_posthoc_dunns(dfs, metric):
    data = [dfs[0][metric], dfs[1][metric], dfs[2][metric]]
    p_vals = posthoc_dunn(data, p_adjust="holm")
    return p_vals

pairwise_groups = ['markov_state', 'markov_gs', 'state_gs']
all_scaled_metrics_dfs = [
    df_scaled_trials_aggregated[:200], 
    df_scaled_trials_aggregated[200:400], 
    df_scaled_trials_aggregated[400:600]
]

krusal_wallis_per_metric = {}
posthoc_dunns_per_metric = {}

anovas_per_metric = {}

for col in metrics_labels:
    # anovas_per_metric[col] = output_anova_results(all_scaled_metrics_dfs, col) #{"f_statistic": f_statistic, "p_value": p_value}
    krusal_wallis_per_metric[col] = output_kruskal_wallis_results(all_scaled_metrics_dfs, col)
    p_val_matrix = compute_posthoc_dunns(all_scaled_metrics_dfs, col)
    # print("PVAL:", p_val_matrix)
    # for row in p_val_matrix:
    #     print(type(row), row)
    #     break
    p_val_flat = flatten_upper_half(p_val_matrix.to_numpy()) # output is indices [(0,1), (0,2), (1,2)]
    # which itself then corresponds to p_val between markov and state, markov and gs, and state and gs
    # print(p_vals)
    posthoc_dunns_per_metric[col] = {
        group: float(p_val_flat[i]) for i, group in enumerate(pairwise_groups)
    }

results_data = {
    # "levine_results": levene_output,
    # "anova_results": anovas_per_metric,
    "kruskal_wallis_results": krusal_wallis_per_metric,
    "posthoc_dunns": posthoc_dunns_per_metric
}

RESULTS_FILE_NAME = f"{TRIAL_NAME}.json"
with open(os.path.normpath(RESULTS_DIR + RESULTS_FILE_NAME), "w") as f:
    json.dump(results_data, f, indent=4)

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Creating + saving box_plots

data_to_plot = [
    markov_flattened_matrix,
    state_flattened_matrix,
    gs_flattened_matrix,
    all_rand_flattened_matrix
]

PLOT_FILE_NAME = f"{len(data_to_plot)}_box_plot.png"
labels = ['Markov \nGroup', 'State-Dependent \nGroup', 'Granular Synthesis \nGroup', 'Baseline Group \n(Fully Randomized)']

plt.figure(figsize=(8, 6))

# Create the boxplot
plt.boxplot(data_to_plot, labels=labels)
plt.ylabel('Pairwise Cosine Distance', fontsize=12)
plt.title('Acoustic Diversity Profile across Parameter Subgroups', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.normpath(FIGURES_DIR + PLOT_FILE_NAME), dpi=300)
# plt.show()

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Creating + saving reduced diminensionality scatter plot of outputs.

print("starting umap process")

PLOT_FILE_PATH_PNG = os.path.normpath(FIGURES_DIR + "umap_acoustic_metrics_space.png") 
PLOT_FILE_PATH_PDF = os.path.normpath(FIGURES_DIR + "umap_acoustic_metrics_space.pdf") 
PLOT_3D_FILE_path_PNG = os.path.normpath(FIGURES_DIR + "umap_3D_acoustic_metrics_space.png") 

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(arr_scaled_trials_aggregated)

len_markov  = 200
len_state   = 200
len_general = 200
len_random  = 400

group_labels = np.repeat(
    ['Markov Group', 'State-Dependent Group', 'Granular Synthesis Group', 'Baseline Group'],
    [len_markov, len_state, len_general, len_random]
)

df_scaled_trials_aggregated['parameter_group'] = group_labels

df_scaled_trials_aggregated['umap_x'] = embedding[:, 0]
df_scaled_trials_aggregated['umap_y'] = embedding[:, 1]

custom_colors = {
    'Markov Group': "#1f77b4",             
    'State-Dependent Group': "#e377c2",    
    'Granular Synthesis Group': '#2ca02c', 
    'Baseline Group': "#7f7f7f"            
}

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='umap_x', y='umap_y', 
    hue='parameter_group', 
    alpha=0.3, 
    palette=custom_colors, 
    data=df_scaled_trials_aggregated
)

plt.title('UMAP Projection of Generative Granular Acoustic Space', fontsize=14, fontweight='bold')
plt.xlabel('UMAP Axis 1')
plt.ylabel('UMAP Axis 2')
plt.legend(title='Parameter Subgroup')
plt.grid(True, alpha=0.3)

plt.savefig(PLOT_FILE_PATH_PDF, format='pdf', bbox_inches='tight')
plt.savefig(PLOT_FILE_PATH_PNG, format='png', dpi=300, bbox_inches='tight')

reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(arr_scaled_trials_aggregated)

df_scaled_trials_aggregated['umap_x'] = embedding[:, 0]
df_scaled_trials_aggregated['umap_y'] = embedding[:, 1]
df_scaled_trials_aggregated['umap_z'] = embedding[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for group, color in custom_colors.items():
    mask = df_scaled_trials_aggregated['parameter_group'] == group
    ax.scatter(
        df_scaled_trials_aggregated.loc[mask, 'umap_x'],
        df_scaled_trials_aggregated.loc[mask, 'umap_y'],
        df_scaled_trials_aggregated.loc[mask, 'umap_z'],
        label=group,
        color=color,
        alpha=0.3
    )

ax.set_title('3D UMAP Projection of Generative Granular Acoustic Space', fontsize=14, fontweight='bold')
ax.set_xlabel('UMAP Axis 1')
ax.set_ylabel('UMAP Axis 2')
ax.set_zlabel('UMAP Axis 3')
ax.legend(title='Parameter Subgroup')

plt.savefig(PLOT_3D_FILE_path_PNG, format='png', dpi=300, bbox_inches='tight')

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Trying HEXBINS instead + scatterplots


HEXBIN_FILE_NAME = os.path.normpath(FIGURES_DIR + "\\PCA_hexbin.png") 
SCATTER_PLOT_FILE_PATH = os.path.normpath(FIGURES_DIR + "\\PCA_scatter_plot.png") 

pca_obj = PCA(n_components=2)
reduced_data = pca_obj.fit_transform(arr_scaled_trials_aggregated)
print(arr_scaled_trials_aggregated.shape, reduced_data.shape)
x = reduced_data.T[0]
y = reduced_data.T[1]

c = ['tab:blue', 'tab:orange', 'tab:green']
data = [[x[:200],y[:200]], [x[200:400],y[200:400]],[x[400:600],y[400:600]],[x[600:],y[600:]]]
get_scatter_plt(file_path=SCATTER_PLOT_FILE_PATH,
             data=data, xlabel="PCA component 1", ylabel="PCA component 2", 
             title="Output 9D metrics PCA components (2) scatter plot", 
             colors=c, labels= labels)


# xlim = x.min(), x.max()
# ylim = y.min(), y.max()

# x_min, x_max = np.percentile(x, [0.5, 99.5])
# y_min, y_max = np.percentile(y, [0.5, 99.5])
# x_pad = (x_max - x_min) * 0.05
# y_pad = (y_max - y_min) * 0.05
# xlim = (x_min - x_pad, x_max + x_pad)
# ylim = (y_min - y_pad, y_max + y_pad)

# fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(11, 4))

# hb0 = ax0.hexbin(x, y, gridsize=20, cmap='inferno', mincnt=1)
# ax0.set(xlim=xlim, ylim=ylim)
# ax0.set_title("Hexagon binning from PCA embeddings of 9D outputs.")
# cb0 = fig.colorbar(hb0, ax=ax0, label='counts')

# hb1 = ax1.hexbin(x, y, gridsize=20, bins='log', mincnt=1, cmap='inferno')
# ax1.set(xlim=xlim, ylim=ylim)
# ax1.set_title("With a log color scale")
# cb1 = fig.colorbar(hb1, ax=ax1, label='log10(counts)', format='$10^{%.1f}$')
# plt.tight_layout()
# plt.savefig(os.path.normpath(FIGURES_DIR + HEXBIN_FILE_NAME + ".png"), format='png', dpi=300, bbox_inches='tight')
