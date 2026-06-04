from datetime import datetime
import hashlib
import json
import os
import soundfile as sf
import numpy as np

def numpy_filler(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_parametre_hashing(param_dict, hash_length):
    """
    get hash based on synthesis params
    """
    param_string = json.dumps(param_dict, default=numpy_filler, sort_keys=True)
    hash_object = hashlib.sha256(param_string.encode())
    full_hash = hash_object.hexdigest()
    return full_hash[:hash_length]
    
def get_output_id(paramdict, hash_length=6):
    """
    use timestamp for YYYY/MM/DD + hash to get metadata and output filepaths
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    param_hash = get_parametre_hashing(paramdict, hash_length)
    filename = f"{timestamp}_{param_hash}"
    return filename + ".wav", filename + ".json"

def save_output_data(output_data, sr, parametre_dict, output_dir, 
                     trial_output_path=None,trial_meta_data_path=None):
    """
    write json metadata and output wav to folders
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    meta_data_path = output_dir + "\metadata"
    output_data_path = output_dir + "\output"
    if trial_output_path:
        output_data_path = trial_output_path
    if trial_meta_data_path:
        meta_data_path = trial_meta_data_path
    if not os.path.exists(meta_data_path):
        os.makedirs(meta_data_path) 
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path, exist_ok=True)
    fp_wav, fp_json = get_output_id(parametre_dict)
    fp_metadata = os.path.join(meta_data_path, fp_json)
    fp_output = os.path.join(output_data_path, fp_wav)
    # print(fp_output)
    with open(fp_metadata, "w") as f:
        json.dump(parametre_dict, f, indent=4)
    if not output_data is None:
        # print(output_data.shape[0])
        sf.write(file=fp_output, data=output_data, samplerate=sr)



