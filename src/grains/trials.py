import numpy as np
from algorithms import MarkovGranulizer
from analysis import AnalyzerObject

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
        })

# Do the analysis of grains in a NB to visualize and choose descriptors

INPUT_PATH =  "..\..\corpus\\pilot_trial_1"
SR = 48000
analyzer = AnalyzerObject(INPUT_PATH, SR)
analyzer.load_y()
grain_duration = 0.1 # 100 ms
grain_size = int(SR*grain_duration)
df = analyzer.compute_grain_descriptors(grain_size)
analyzer.save_metadata(df, grain_duration=grain_duration)
METADATA_PATH = "..\..\corpus\\flute_sample_1\metadata\grain_0.14_s_metadata_e118eb9b.csv" 
df = analyzer.load_metadata(metadata_path)
_, df_scaled = analyzer.scale_metadata(df, scaler=2)
x = "rolloff"
y = "crest"
features = [x,y]
# df
# Do the trial runs

for trial in range(N_CONFIGURATIONS):



