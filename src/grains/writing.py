from datetime import datetime
import hashlib
import json
import os
import soundfile


def get_parametre_hashing(param_dict, hash_length):
    """
    get hash based on synthesis params
    """
    param_string = json.dumps(param_dict, sort_keys=True)
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

def save_output_data(output_data, sr, parametre_dict, output_dir):
    """
    write json metadata and output wav to folders
    """
    fp_wav, fp_json = get_output_id(parametre_dict)
    fp_metadata = os.path.join((output_dir + "\metadata"), fp_json)
    fp_output = os.path.join((output_dir + "\output"), fp_wav)
    with open(fp_metadata, "w") as f:
        json.dump(parametre_dict, f, indent=4)
    sf.write(fp_output, output_data, sr)
    