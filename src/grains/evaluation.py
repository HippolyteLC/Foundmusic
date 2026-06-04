import json


### 1) collect metrics from outputs 
# - random seeds
# - parametre configurations
# - output metrics 
# - trial block
# segment first 300 as Markov
# segment second 300 as State
# segment third 300 as General
# segment last 600 as All
general_params_path = "..\..\corpus\pilot_study_2\\trial_data\params\\20260604_183256_trial.json"
with open(general_params_path, "r") as f:
    data = json.load(f)

### 2) com