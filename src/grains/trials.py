import numpy as np
from algorithms import MarkovGranulizer, rand_tpm
from analysis import AnalyzerObject
from helpers import rev_exp
import writing
import json 
import os

# Set the seeds

N_CONFIGURATIONS = 10
K_REPETITIONS = 5

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
            'cluster_sampling_seed': cluster_sampling_seeds[j]
            'grain_position_sampling_seeds': grain_position_sampling_seeds[j]
        })

# Do the analysis of grains in a NB to visualize and choose descriptors

PATH =  "..\..\corpus\\pilot_trial_1"
SR = 48000
analyzer = AnalyzerObject(PATH, SR)
analyzer.load_y()
grain_duration = 0.1 # 100 ms
grain_size = int(SR*grain_duration)
df = analyzer.compute_grain_descriptors(grain_size)
analyzer.save_metadata(df, grain_duration=grain_duration)
_, df_scaled = analyzer.scale_metadata(df, scaler=2)
x = "rolloff"
y = "crest"
features = [x,y]


# Set the parametre value ranges

grains = analyzer.grains()
n_streams_arr = [1,2,4,16]
windows = [np.hanning, rev_exp] # add exp
density_arrays = [
    [1, 2, 4]
    [10, 20, 40],
    [100, 200, 400]
]
grain_sizes = [
    [1, 10, 20],
    [40, 60, 100]
]
n_clusters_arr = [2, 3, 5, 8]

granulator = MarkovGranulizer(sr=SR, grain_size=grain_size)

# Do the trial runs

for trial in range(N_CONFIGURATIONS):
    config_seed = trials[trial]["config_seed"]
    param_config_rng = np.random.default_rng(config_seed)
   
    n_clusters = param_config_rng.choice(n_clusters_arr) 
    kmeans_obj = analyzer.compute_kmeans(df_scaled, n_clusters=n_clusters, features=features)
    dict_clusters, _ = analyzer.get_cluster_dict(kmeans_obj.labels_)
    
    n_states = len(grain_sizes) * len(densities) * n_clusters
    

    # randomize params per trial
    tpm = rand_tpm(n_states, config_seed)
    init_states = [param_config_rng.randint(0, n_states) for _ in range(n_streams)]
    densities = [param_config_rng.choice(i) for i in density_arrays]
    window = param_config_rng.choice(windows)
    n_streams = param_config_rng.choice(n_streams_arr)

    for rep in range(K_REPETITIONS):

        audio_arr, markchains, params = granulator.run_v3(
            y=analyzer.y,
            densities=densities,
            grain_sizes=grain_sizes,
            init_states=init_states,
            tpm=tpm,
            dict_clusters=dict_clusters,
            grains=grains,
            seed_grain_sampling=trials[rep]["grain_sampling_seed"],
            seed_state_sampling=trials[rep]["state_sampling_seed"],
            seed_grain_pos_sampling=trials[rep]["grain_pos_sampling_seed"],
        )
        output_analyzer = AnalyzerObject(PATH, SR)
        spec_arr, spectral_obj = output_analyzer.get_spectral_arr(y=audio_arr)
        centroid_arr = spectral_obj.centroid(spec_arr)
        flux_arr = spectral_obj.flux(spec_arr)
        trial_id = trials[trial]["config_id"]
        repetition_id = trials[trial]["rep_id"]

        params["trial_id"] = trial_id
        params["rep_id"] = repetition_id
        params["centroid_arr"] = centroid_arr.tolist()
        params["flux_arr"] = flux_arr.tolist()
        params["markov_chains"] = markchains

        # TODO: save every 10th output
        if trial % 10:
            writing.save_output_data(
                output_data=audio_arr,
                sr=SR,
                parametre_dict=params,
                output_dir=PATH
                )
        trial_data_path = os.path.normpath(PATH + "\\trial_data")
        if not os.path.exists
        