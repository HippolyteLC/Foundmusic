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
from analysis import get_histograms, get_scatter_plt, get_spectrogram, get_density_trellis
from scikit_posthocs import posthoc_dunn
import audioflux as af
from sklearn.preprocessing import normalize

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### IMPORTANT: every unqiue config's K repetitions are aggregated and the mean is taken
# This is done to keep the iid property for statistical testing of variety of outputs

# Some booleans to determine which plots/ statistics are produced.
do_UMAP_scatterplot = False
do_boxplot = False
do_violin_plot_sns = False
do_violin_plot_plt = True
do_cosine_distance_comp = False if not (do_violin_plot_sns or do_violin_plot_plt) else True

STUDY_NAME = "pilot_study_3"
TRIAL_NAME = "20260611_182323_trial"
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
    SR = data_dict["general_trial_parametres"]["sr"]
    # N_CONFIGS_PER_PARAM_GROUP = data_dict["general_trial_parametres"][""]
    # TODO: added dynamic loading of configs per group

### Analyzing the input 
TRIAL_INPUT = f"..\..\corpus\{STUDY_NAME}\\input.wav"
INPUT_SPECTROGRAM_PATH = os.path.normpath(FIGURES_DIR + "input_spectrogram.pdf")
y,_ = af.read(TRIAL_INPUT, samplate=SR)
get_spectrogram(INPUT_SPECTROGRAM_PATH,y,SR)

all_trials=[]
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
    data = np.array(df)
    n_samples = data.shape[0]
    for i in range(n_samples):
        row = []
        for j in range(n_samples):
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
# print(arr_scaled_trials_aggregated.shape)


if do_cosine_distance_comp:
    print(arr_scaled_trials_aggregated[:200].shape)
    l2_normalized_data = normalize(arr_scaled_trials_aggregated)
    print(l2_normalized_data.shape)
    markov_flattened_matrix = flatten_upper_half(compute_cosine_matrix(l2_normalized_data[:200]))
    state_flattened_matrix = flatten_upper_half(compute_cosine_matrix(l2_normalized_data[200:400]))
    gs_flattened_matrix = flatten_upper_half(compute_cosine_matrix(l2_normalized_data[400:600]))
    all_rand_flattened_matrix = flatten_upper_half(compute_cosine_matrix(l2_normalized_data[:600]))
    print(markov_flattened_matrix.shape)
###_____________________________________________________________________________###
###_____________________________________________________________________________###
### KDE plots - analyze data
# TODO: produce histograms to display the 9 output metrics per parametre subgroup. 
labels = ['Markov \nGroup', 'State \nGroup', 'GS \nGroup', 'Baseline Group \n(Fully Randomized)']
metrics_labels = list(df_scaled_trials_aggregated.columns)

get_histograms(FIGURES_DIR, "markov_outputs_histograms.pdf",df_scaled_trials_aggregated[:200], metrics_labels)
get_histograms(FIGURES_DIR, "state_outputs_histograms.pdf", df_scaled_trials_aggregated[200:400], metrics_labels)
get_histograms(FIGURES_DIR, "gs_outputs_histograms.pdf", df_scaled_trials_aggregated[400:600], metrics_labels)
# get_histograms(FIGURES_DIR, "all_random_outputs_histograms_.png", df_scaled_trials_aggregated[:600], metrics_labels)
all_scaled_metrics_dfs = [
    df_scaled_trials_aggregated[:200], 
    df_scaled_trials_aggregated[200:400], 
    df_scaled_trials_aggregated[400:600]
]

get_density_trellis(FIGURES_DIR, "KDE_plots_output_distributions.pdf", all_scaled_metrics_dfs,
                    labels[:3],metrics_labels)

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Get results from cosine_dist metrics (ANOVA + Levine)

def format_p_value(p):
    return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"

def compute_kruskal_wallis_one_way(dfs, metric):
    H_statistic, p_value = stats.kruskal(dfs[0][metric], dfs[1][metric], dfs[2][metric])
    return H_statistic, p_value

def output_kruskal_wallis_results(dfs, metric):
    H_statistic, p_value = compute_kruskal_wallis_one_way(dfs, metric)
    return {"H_stat": H_statistic, "p_val": p_value}

def compute_posthoc_dunns(dfs, metric):
    data = [dfs[0][metric], dfs[1][metric], dfs[2][metric]]
    p_vals = posthoc_dunn(data, p_adjust="holm")
    return p_vals

pairwise_groups = ['markov_state', 'markov_gs', 'state_gs']
# all_scaled_metrics_dfs = [
#     df_scaled_trials_aggregated[:200], 
#     df_scaled_trials_aggregated[200:400], 
#     df_scaled_trials_aggregated[400:600]
# ]

krusal_wallis_per_metric = {}
posthoc_dunns_per_metric = {}

for col in metrics_labels:
    krusal_wallis_per_metric[col] = output_kruskal_wallis_results(all_scaled_metrics_dfs, col)
    p_val_matrix = compute_posthoc_dunns(all_scaled_metrics_dfs, col)
    p_val_flat = flatten_upper_half(p_val_matrix.to_numpy()) # output is indices [(0,1), (0,2), (1,2)]
    posthoc_dunns_per_metric[col] = {
        group: float(p_val_flat[i]) for i, group in enumerate(pairwise_groups)
    }   

results_data = {
    "kruskal_wallis_results": krusal_wallis_per_metric,
    "posthoc_dunns": posthoc_dunns_per_metric
}

kruskal_df = pd.DataFrame(results_data['kruskal_wallis_results']).T
kruskal_df.to_csv(os.path.normpath(RESULTS_DIR + f'{TRIAL_NAME}_kruskal_results.csv'))

posthoc_df = pd.DataFrame(results_data['posthoc_dunns']).T
posthoc_df.to_csv(os.path.normpath(RESULTS_DIR + f'{TRIAL_NAME}_posthoc_results.csv'))


RESULTS_FILE_NAME = f"{TRIAL_NAME}.json"
with open(os.path.normpath(RESULTS_DIR + RESULTS_FILE_NAME), "w") as f:
    json.dump(results_data, f, indent=4)

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Creating + saving box_plots

if do_cosine_distance_comp:
    data_to_plot = [
        markov_flattened_matrix,
        state_flattened_matrix,
        gs_flattened_matrix,
        all_rand_flattened_matrix
    ]


    statistics = {}
    for idx, data in enumerate(data_to_plot):
        statistics[labels[idx]] = {
            "median": float(np.median(data)),
            "iqr": float(stats.iqr(data))
        }
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(os.path.normpath(RESULTS_DIR + f'{TRIAL_NAME}_statistics_cosine_results.csv'))

BOX_PLOT_FILE_NAME = f"box_plot.pdf"
VIOLIN_PLOT_PLT_FILE_NAME = f"violin_plot_plt.pdf"
VIOLIN_PLOT_SNS_FILE_NAME = f"violin_plot_sns.pdf"

if do_boxplot == True:

    plt.figure(figsize=(8, 6))

    # Create the boxplot
    plt.boxplot(data_to_plot, labels=labels)

    plt.ylabel('Pairwise Cosine Distance', fontsize=12)
    plt.title('Acoustic Diversity across Parameter Subgroups', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.normpath(FIGURES_DIR + BOX_PLOT_FILE_NAME), dpi=300, format='pdf')
    plt.close()

    plt.figure(figsize=(8, 6))

if do_violin_plot_plt:
    # sns.violinplot(data=data_to_plot, color="black") TODO
    plt.violinplot(data_to_plot, showmedians=True)
    plt.xticks(ticks=np.arange(1, len(data_to_plot) + 1), labels=labels)

    plt.ylabel('Pairwise Cosine Distance', fontsize=12)
    plt.title('Acoustic Diversity across Parameter Subgroups', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.normpath(FIGURES_DIR + VIOLIN_PLOT_PLT_FILE_NAME), dpi=300, format='pdf')
    plt.close()


if do_violin_plot_sns:
    # sns violin + swarm plot 
    plt.figure(figsize=(8, 6))

    sns.violinplot(data=data_to_plot, inner="quart", color="salmon")
    # sns.swarmplot(data=data_to_plot, color="maroon", size=3)

    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.ylabel('Pairwise Cosine Distance', fontsize=12)
    plt.title('Acoustic Diversity across Parameter Subgroups', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(os.path.normpath(FIGURES_DIR + VIOLIN_PLOT_SNS_FILE_NAME), dpi=300, format='pdf')
    plt.close()

    print([data_to_plot[i].shape for i in range(len(data_to_plot)) ])

###_____________________________________________________________________________###
###_____________________________________________________________________________###
### Creating + saving reduced diminension scatter plot of outputs.

if do_UMAP_scatterplot == True:
    print("starting umap process")

    PLOT_FILE_PATH_PDF = os.path.normpath(FIGURES_DIR + "umap_acoustic_metrics_space.pdf") 

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.75, random_state=42)
    # increased min dist as clusters are too tight
    embedding = reducer.fit_transform(arr_scaled_trials_aggregated)

    len_markov  = 200
    len_state   = 200
    len_general = 200
    len_random  = 400

    group_labels = np.repeat(
        ['Markov Group', 'State Group', 'GS Group', 'Baseline Group'],
        [len_markov, len_state, len_general, len_random]
    )

    umap_df = pd.DataFrame({
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1], 
        "group_labels": group_labels
    })

    custom_colors = {
        'Markov Group': "#1f81d0",             
        'State Group': "#d89e20",    
        'GS Group': "#24de13", 
        'Baseline Group': "#89b0e6"            
    }

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='umap_x', y='umap_y', 
        alpha=0.8, 
        hue="group_labels",
        palette=custom_colors, 
        data=umap_df
    )

    plt.title('UMAP Projection of Output Acoustic Space', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Axis 1')
    plt.ylabel('UMAP Axis 2')
    plt.legend(title='Parameter Subgroup')
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_FILE_PATH_PDF, format='pdf', bbox_inches='tight')
    plt.close()

