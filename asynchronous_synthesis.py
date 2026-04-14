import audioflux as af
from audioflux.type import WindowType
import subprocess
import os
import numpy as np
import csv

# 1: get slices 
FILE_PATH = r"corpus\metro_sample_2.wav"
OUTPUT_SLICE_PATH = os.path.join("slicing_index/", FILE_PATH)

command = [
    "fluid-onsetslice",
    "-source", FILE_PATH,
    "-indices", OUTPUT_SLICE_PATH,
    "-metric", "9" # rectified phase dev. How much the next spectral image differs from the anticipated prior
]
try:
    result = subprocess.run(
        command, 
        check=True,          
        capture_output=True, 
        text=True            
    )
except subprocess.CalledProcessError as e:
    print("Error details:", e.stderr)
with open(OUTPUT_SLICE_PATH, 'w') as f:
    features = np.array([int(i) for i in list(csv.reader(f))[0]])   
  

# 1.1: load audio
audio_arr, sr = af.read(FILE_PATH)
radix2_exp = 11
slide_length = (1 << radix2_exp) // 4
hpss_obj = af.HPSS(radix2_exp=radix2_exp, window_type=WindowType.HAMM,
                   slide_length=slide_length, h_order=21, p_order=31)
harmonic_arr, p_arr = hpss_obj.hpss(audio_arr)


# 2: analyze all slices 



# 3: asynchronous synthesis