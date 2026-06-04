import numpy as np
from algorithms import MarkovGranulizer, rand_tpm
from analysis import AnalyzerObject
from helpers import rev_exp
import writing
import json 
import os
from datetime import datetime

# Set the seeds

N_CONFIGURATIONS = 1
K_REPETITIONS = 2

master_rng = np.random.default_rng(42)

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

# Do the analysis of grains in a NB to visualize and choose descriptors

PATH =  "..\..\corpus\\pilot_trial_1"
SR = 48000

time = datetime.now().strftime("%Y%m%d_%H%M%S")
trial_dir = os.path.normpath(PATH + f"\\trial_data\\")
trial_path = os.path.normpath(PATH + f"\\trial_data\\{time}_trial.json")
if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)

analyzer = AnalyzerObject(PATH, SR)
analyzer.load_y()
# print(analyzer.y.shape)
grain_duration = 0.1 # 100 ms
grain_size = int(SR*grain_duration)
METADATA_PATH = "..\..\corpus\pilot_trial_1\metadata\grain_0.1_s_metadata_6d91620b.csv"
if os.path.exists(METADATA_PATH):
    df = analyzer.load_metadata(METADATA_PATH)
else:
    df = analyzer.compute_grain_descriptors(grain_size)
    analyzer.save_metadata(df, grain_duration=grain_duration)
_, df_scaled = analyzer.scale_metadata(df, scaler=2)
x = "rolloff"
y = "crest"
features = [x,y]


# Set the parametre value ranges

grains = analyzer.grains(grain_size)
n_streams_arr = [1,2,4,16]
windows = [np.hanning, rev_exp] # add exp
density_arrays = [
    [1, 2, 4],
    [10, 20, 40],
    [100, 200, 400]
]
grain_size_arrays = [
    [1, 10, 20],
    [40, 60, 100]
]
n_clusters_arr = [2, 3, 5, 8]

granulator = MarkovGranulizer(sr=SR)

# Do the trial runs

print("starting trials")
for trial in range(N_CONFIGURATIONS):
    print(f"trial {trial}")
    config_seed = trials[trial]["config_seed"]
    param_config_rng = np.random.default_rng(config_seed)
   
    n_clusters = int(param_config_rng.choice(n_clusters_arr)) 
    kmeans_obj = analyzer.compute_kmeans(df_scaled, n_clusters=n_clusters, features=features)
    dict_clusters = analyzer.get_cluster_dict(kmeans_obj.labels_)

    n_states = len(grain_size_arrays) * len(density_arrays) * n_clusters
    print(f"n states: {n_states} , clusters {n_clusters}")

    # randomize params per trial
    densities = [int(param_config_rng.choice(i)) for i in density_arrays]
    grain_sizes = [int(param_config_rng.choice(i)) for i in grain_size_arrays]

    window = param_config_rng.choice(windows)
    n_streams = int(param_config_rng.choice(n_streams_arr))

    tpm = rand_tpm(n_states, config_seed)
    # init_states = [param_config_rng.integers(0, n_states) for _ in range(n_streams)]

    for rep in range(K_REPETITIONS):

        audio_arr, markchains, params = granulator.run_v3(
            y=analyzer.y,
            densities=densities,
            grain_size=grain_size,
            grains=grains,
            grain_sizes=grain_sizes,
            tpm=tpm,
            dict_clusters=dict_clusters,
            n_streams=n_streams,
            window=window,
            n_clusters=n_clusters,
            config_seed=config_seed,
            seed_grain_sampling=trials[rep]["cluster_sampling_seed"],
            seed_state_sampling=trials[rep]["state_sampling_seed"],
            seed_grain_pos_sampling=trials[rep]["grain_position_sampling_seed"],
        )
        output_analyzer = AnalyzerObject(PATH, SR)
        spec_arr, spectral_obj = output_analyzer.get_spectral_arr(y=audio_arr)
        centroid_arr = spectral_obj.centroid(spec_arr)
        flux_arr = spectral_obj.flux(spec_arr)
        trial_id = trials[trial]["config_id"]
        repetition_id = trials[trial]["rep_id"]
        # print(flux_arr.shape)
        params["sr"] = SR
        params["trial_id"] = trial_id
        params["rep_id"] = repetition_id
        params["centroid_arr"] = [float(i) for i in centroid_arr.tolist()]
        params["flux_arr"] = [float(i) for i in flux_arr.tolist()]
        params["markov_chains"] = [int(i) for i in markchains[0]]
        
        # print(list((k, type(v)) for k, v in params.items()))
        # print(params)
        print("output generared and params collected!")
        # break
        
        # Logging + saving output data
        if trial % 10 == 0:
            writing.save_output_data(
                output_data=audio_arr,
                sr=SR,
                parametre_dict=params,
                output_dir=PATH
                )
        print("trial output saved")



        if os.path.exists(trial_path):
            with open(trial_path, 'r') as f:
                all_trials = json.load(f)
        else:
            all_trials = [] 
        print("checked trial data save path exists")

        all_trials.append(params)
        
        with open(trial_path, 'w') as f:
            json.dump(all_trials, f) 

        print(f'saved trial {trial_id}')