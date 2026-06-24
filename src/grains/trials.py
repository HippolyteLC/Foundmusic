import numpy as np
import pandas as pd
from grains.algorithms import MarkovGranulizer, rand_tpm
from grains.analysis import AnalyzerObject, get_spectrogram
from grains.helpers import expodec, rexpodec, sinc_envelope
import grains.writing
import json 
import os
from datetime import datetime
from scipy import stats
import warnings
import umap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')

# directory + sample rate
STUDY_NAME = "pilot_study_3"
PATH =  f"..\..\corpus\\{STUDY_NAME}"
SR = 48000

# metadata path of input.wav for specific grain duration
GRAIN_DURATION = 0.1 # 100 ms
METADATA_PATH = f"..\..\corpus\{STUDY_NAME}\metadata\grain_{GRAIN_DURATION}_s_metadata_e8f6c3ba.csv"

### Modify parametre ranges below. 
# to change the order or ranges of specific subgroup tests change below.

# Set the seeds
N_CONFIGS_PER_PARAMGROUP = 200
N_PARAM_GROUPS = 3
N_CONFIGS_ALL_RAND = 400
N_CONFIGURATIONS = (N_CONFIGS_PER_PARAMGROUP * N_PARAM_GROUPS) + N_CONFIGS_ALL_RAND
K_REPETITIONS = 5

MASTER_SEED = 42

# grain analysis feature array
# if set to None (COMMENT OUT), all the features available will be used to compute KMeans
# features = ["rolloff", "crest"] # default
features = None

master_rng = np.random.default_rng(MASTER_SEED)

trials = []
state_sampling_seeds = []
cluster_sampling_seeds = []
grain_position_sampling_seeds = []

for i in range(K_REPETITIONS):
    state_sampling_seeds.append(int(master_rng.integers(0, 99999)))
    cluster_sampling_seeds.append(int(master_rng.integers(0, 99999)))
    grain_position_sampling_seeds.append(int(master_rng.integers(0, 99999)))

for i in range(N_CONFIGURATIONS):
    config_seed = int(master_rng.integers(0, 99999))
    for j in range(K_REPETITIONS):
        trials.append({
            'config_id':             i,
            'rep_id':                j,
            'config_seed':           config_seed,  
            'state_sampling_seed':   state_sampling_seeds[j],
            'cluster_sampling_seed': cluster_sampling_seeds[j],
            'grain_position_sampling_seed': grain_position_sampling_seeds[j]
        })


time = datetime.now().strftime("%Y%m%d_%H%M%S")
trial_dir = os.path.normpath(PATH + f"\\trial_data\\")
trial_logs_dir = os.path.normpath(trial_dir + "\\logs")
trial_params_dir = os.path.normpath(trial_dir + "\\params")
trial_figures_dir = os.path.normpath(trial_dir + "\\figures")
trial_outputs_dir = os.path.normpath(trial_dir + "\\outputs" + f"\\{time}_trial")
trial_metadata_dir = os.path.normpath(trial_dir + "\\metadata" + f"\\{time}_trial")

if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)
if not os.path.exists(trial_logs_dir):
    os.makedirs(trial_logs_dir)
if not os.path.exists(trial_params_dir):
    os.makedirs(trial_params_dir)
if not os.path.exists(trial_outputs_dir):
    os.makedirs(trial_outputs_dir)
if not os.path.exists(trial_metadata_dir):
    os.makedirs(trial_metadata_dir)

trial_logs_path = os.path.normpath(trial_logs_dir + f"\\{time}_trial.jsonl")
trial_params_path = os.path.normpath(trial_params_dir + f"\\{time}_trial.json")

analyzer = AnalyzerObject(PATH, SR)
analyzer.load_y()
y = analyzer.y
# if len(y) > SR*10:
#     y = y[:SR*10]
grain_size = int(SR*GRAIN_DURATION)

if os.path.exists(METADATA_PATH):
    df = analyzer.load_metadata(METADATA_PATH)
    df = df.iloc[:, 4:]
else:
    df = analyzer.compute_grain_descriptors(grain_size)
    analyzer.save_metadata(df, grain_duration=GRAIN_DURATION)
_, df_scaled = analyzer.scale_metadata(df, scaler=2)

grains = analyzer.grains(grain_size)
granulator = MarkovGranulizer(sr=SR)
output_analyzer = AnalyzerObject(PATH, SR)

### Get input spectrogram + grain distribution (UMAP)
input_spectrogram_path = os.path.normpath(trial_figures_dir + "\\input_spectrogram.pdf")
print("PATH:", input_spectrogram_path)
if not os.path.exists(input_spectrogram_path):
    get_spectrogram(input_spectrogram_path, y, SR)

input_grain_analysis_plot = os.path.normpath(trial_figures_dir + "\\input_grain_analysis.pdf")
if not os.path.exists(input_grain_analysis_plot):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=MASTER_SEED)
    embedding = reducer.fit_transform(df_scaled)
    umap_df = pd.DataFrame(embedding, columns=['umap_x', 'umap_y'])
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='umap_x', y='umap_y', 
        alpha=0.5, 
        data=umap_df
    )

    plt.title('UMAP Projection of Grains Descriptor Space', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Axis 1')
    plt.ylabel('UMAP Axis 2')
    plt.grid(True, alpha=0.5)

    plt.savefig(input_grain_analysis_plot, format='pdf', dpi=300, bbox_inches='tight')
    plt.close() 

### Set the parametre value ranges

# n_streams_arr = [1,2,4,16]
# windows = [np.hanning, expodec, rexpodec, sinc_envelope] # add exp
# density_arrays = [
#     [1, 1, 1], # uniform low
#     [1, 3, 9], # low increasing (x3)
#     [10,10,10], # uniform medium 
#     [10,30,90], # medium increasing (x3)
#     [100,100,100], # uniform high
#     [100, 300, 900], # high increasing (x3)
#     [1, 90, 900], # high contrast low-high
#     [30, 90, 100] # low contrast medium-high
#     # [1, 10, 100, 1000],
#     # [3, 30, 300, 3000],
#     # [9, 90, 900, 9000]
# ]
# grain_duration_arrays = [
#     [1,1,1], # low only
#     [40, 40, 40],# medium only
#     [100,100,100], # high only
#     [1, 40, 100], # increasing (even sampling)
#     [100,1,100] # high contrast
# ]
# grain_size_arrays = [ 
#     [int(i*SR//1000) for i in j] for j in grain_duration_arrays
# ]

# n_clusters_arr = [2, 3, 5, 8]

# general_trial_config = {
#     "features": features,
#     "n_configurations": N_CONFIGURATIONS,
#     "k_repetitions": K_REPETITIONS,
#     "master_seed": MASTER_SEED,
#     "sr": SR,
#     "grain_size": grain_size
# }
# config_param_value_ranges = {
#     "n_streams_arr": n_streams_arr,
#     "windows": [str(win.__name__) for win in windows],
#     "density_arrays": density_arrays,
#     "grain_size_arrays": grain_size_arrays,
#     "n_clusters_arr": n_clusters_arr
# }

# general_config = {
#     "general_trial_parametres": general_trial_config,
#     "config_parametre_ranges": config_param_value_ranges
# }

# with open(trial_params_path, "w") as f:
#     json.dump(general_config, f, indent=4)

# ### Refactor into funcs

# def get_metrics(audio_arr):
#     spec_arr, spectral_obj = output_analyzer.get_spectral_arr(y=audio_arr)
#     flatness_arr = spectral_obj.flatness(spec_arr)
#     flux_arr = spectral_obj.flux(spec_arr)
#     centroid_arr = spectral_obj.centroid(spec_arr)
#     metrics = {
#         "centroid_mean": np.mean(centroid_arr),
#         "centroid_std": np.std(centroid_arr),
#         "centroid_skewness": float(np.nan_to_num(stats.skew(centroid_arr), nan=0.0)),
#         "flatness_mean": np.mean(flatness_arr),
#         "flatness_std": np.std(flatness_arr),
#         "flatness_skewness": float(np.nan_to_num(stats.skew(flatness_arr), nan=0.0)),
#         "flux_mean": np.mean(flux_arr),
#         "flux_std": np.std(flux_arr),
#         "flux_skewness": float(np.nan_to_num(stats.skew(flux_arr), nan=0.0))
#     }
#     return metrics

# ### Do the trial runs

# ##### Trial 1,2, and 3 (Sub group studies)
# # Trial 1: Markov, Trial 2: State, Trial 3: General

# print(f"starting trials Trial 1,2, and 3 (Sub group studies)")
# print("Trial 1: Markov, Trial 2: State, Trial 3: General")

# last_n_clusters = None
# # all_trials = []
# for trial in range(N_CONFIGS_PER_PARAMGROUP * N_PARAM_GROUPS * K_REPETITIONS):
#     config_seed = trials[trial]["config_seed"]
#     config_id = trials[trial]["config_id"]
#     repetition_id = trials[trial]["rep_id"]
#     trial_id = f"config_{config_id}_rep_{repetition_id}"
#     if trial % 50 == 0:
#         print(f"trial: {trial_id}")

#     # STATE parametres
#     if N_CONFIGS_PER_PARAMGROUP <= config_id < N_CONFIGS_PER_PARAMGROUP*2:
#         param_config_rng = np.random.default_rng(config_seed) # unfrozen seed

#         densities = [int(i) for i in param_config_rng.choice(density_arrays).tolist()]
#         grain_sizes = [int(i) for i in param_config_rng.choice(grain_size_arrays).tolist()]
#         n_clusters = int(param_config_rng.choice(n_clusters_arr)) 
#     else:
#         param_config_rng = np.random.default_rng(trials[0]["config_seed"]) # unfrozen seed

#         densities = [int(i) for i in param_config_rng.choice(density_arrays)]
#         grain_sizes = [int(i) for i in param_config_rng.choice(grain_size_arrays)]
#         n_clusters = int(param_config_rng.choice(n_clusters_arr)) 

#     if not last_n_clusters == n_clusters:
#         kmeans_obj = analyzer.compute_kmeans(df_scaled, n_clusters=n_clusters, features=features)
#         dict_clusters = analyzer.get_cluster_dict(kmeans_obj.labels_)
#         n_states = len(grain_sizes) * len(densities) * n_clusters

#     last_n_clusters = n_clusters

#     # GS parametres
#     if N_CONFIGS_PER_PARAMGROUP*2 <= config_id < N_CONFIGS_PER_PARAMGROUP*3: 
#         param_config_rng = np.random.default_rng(config_seed) # unfrozen seed

#         window = param_config_rng.choice(windows)
#         n_streams = int(param_config_rng.choice(n_streams_arr))
#     else:
#         param_config_rng = np.random.default_rng(trials[0]["config_seed"]) # unfrozen seed

#         window = param_config_rng.choice(windows)
#         n_streams = int(param_config_rng.choice(n_streams_arr))

#     # MARKOV parametre
#     if config_id < N_CONFIGS_PER_PARAMGROUP:
#         conf_seed = config_seed
#         tpm = rand_tpm(n_states, conf_seed)
#     else:       
#         conf_seed = trials[0]["config_seed"]
#         tpm = rand_tpm(n_states, conf_seed)        

#     audio_arr, markchains, params = granulator.run_v3(
#         y=y,
#         densities=densities,
#         grain_size=grain_size,
#         grains=grains,
#         grain_sizes=grain_sizes,
#         tpm=tpm,
#         dict_clusters=dict_clusters,
#         n_streams=n_streams,
#         window=window,
#         n_clusters=n_clusters,
#         config_seed=conf_seed,
#         seed_grain_sampling=trials[trial]["cluster_sampling_seed"],
#         seed_state_sampling=trials[trial]["state_sampling_seed"],
#         seed_grain_pos_sampling=trials[trial]["grain_position_sampling_seed"],
#     )
#     synthesis_parametres = params
#     parametres = {}
#     parametres["trial_id"] = trial_id
#     parametres["synthesis_params"] = params

#     metrics = get_metrics(audio_arr)

#     parametres["metrics"] = {k: float(v) for k,v in metrics.items()}
#     parametres["markov_chains"] = [[int(i) for i in j] for j in markchains]
        
#     # Logging + saving output data
#     if trial % 250 == 0:
#         writing.save_output_data(
#             output_data=audio_arr,
#             sr=SR,
#             parametre_dict=parametres,
#             output_dir=PATH,
#             trial_output_path=trial_outputs_dir,
#             trial_meta_data_path=trial_metadata_dir
#             )
    
#     with open(trial_logs_path, 'a') as f:
#         f.write(json.dumps(parametres) + "\n")


# ##### Trial 4 (All parametres unfrozen)
# last_n_clusters = None


# print(f"starting trials Trial 4 (All parametres unfrozen)")

# for trial in range(N_CONFIGS_PER_PARAMGROUP * N_PARAM_GROUPS * K_REPETITIONS, N_CONFIGURATIONS * K_REPETITIONS):
#     config_seed = trials[trial]["config_seed"]
#     config_id = trials[trial]["config_id"]
#     repetition_id = trials[trial]["rep_id"]
#     trial_id = f"config_{config_id}_rep_{repetition_id}"
#     if trial % 50 == 0:
#         print(f"trial: {trial_id}")

#     # STATE parametres
#     param_config_rng = np.random.default_rng(config_seed) # unfrozen seed

#     densities = [int(i) for i in param_config_rng.choice(density_arrays).tolist()]
#     grain_sizes = [int(i) for i in param_config_rng.choice(grain_size_arrays).tolist()]
#     n_clusters = int(param_config_rng.choice(n_clusters_arr)) 

#     if not last_n_clusters == n_clusters:
#         kmeans_obj = analyzer.compute_kmeans(df_scaled, n_clusters=n_clusters, features=features)
#         dict_clusters = analyzer.get_cluster_dict(kmeans_obj.labels_)
#         if len(dict_clusters) < n_clusters:
#             print(f"KMeans produced empty clusters. Set n_clusters={n_clusters}, got {len(dict_clusters)} clusters.")
#         n_states = len(grain_sizes) * len(densities) * n_clusters

#     last_n_clusters = n_clusters

#     # GS parametres
#     window = param_config_rng.choice(windows)
#     n_streams = int(param_config_rng.choice(n_streams_arr))

#     # MARKOV parametre
#     conf_seed = config_seed
#     tpm = rand_tpm(n_states, conf_seed)
  
#     audio_arr, markchains, params = granulator.run_v3(
#         y=y,
#         densities=densities,
#         grain_size=grain_size,
#         grains=grains,
#         grain_sizes=grain_sizes,
#         tpm=tpm,
#         dict_clusters=dict_clusters,
#         n_streams=n_streams,
#         window=window,
#         n_clusters=n_clusters,
#         config_seed=conf_seed,
#         seed_grain_sampling=trials[trial]["cluster_sampling_seed"],
#         seed_state_sampling=trials[trial]["state_sampling_seed"],
#         seed_grain_pos_sampling=trials[trial]["grain_position_sampling_seed"],
#     )
#     synthesis_parametres = params
#     parametres = {}
#     parametres["trial_id"] = trial_id
#     parametres["synthesis_params"] = params
    
#     metrics = get_metrics(audio_arr)

#     parametres["metrics"] = {k: float(v) for k,v in metrics.items()}
#     parametres["markov_chains"] = [[int(i) for i in j] for j in markchains]
    

#     # print("output generared and params collected!")
    
#     # Logging + saving output data
#     if trial % 250 == 0:
#         writing.save_output_data(
#             output_data=audio_arr,
#             sr=SR,
#             parametre_dict=parametres,
#             output_dir=PATH,
#             trial_output_path=trial_outputs_dir,
#             trial_meta_data_path=trial_metadata_dir
#             )
        
#     with open(trial_logs_path, 'a') as f:
#         f.write(json.dumps(parametres) + "\n")