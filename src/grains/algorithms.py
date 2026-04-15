import numpy as np
import soundfile as sf

def async_gs(input_path, output_file_dir, data, grain_indices, 
             sr, cloud_duration, n_streams, 
             max_grain_density, rand_seed=None) -> None:
    """
    This function takes several grain parametres as input and produces some 
    asynchronous cloud by adding several streams of grains with each other. The 
    produced audio array is saved as a wav file. 

    This is an unfinished implementation. Typically, you would precompute an array with the 
    indices of the grains from the original slicing data. 

    input_path: string representing input.wav file location
    output_file_dir: string representing outout *.wav file location
    data: array containing the slice indices of grains in original array
    grain_indices: array containing the indices of selected grain in the data array
    sr: samplerate
    cloud_duration: duration of an asynchronous cloud
    n_streams: number of streams, where each stream has one unique grain from which it samples
    max_grain_density: maximum grain density per stream in Hz (per second)
    rand_seed: np random seed
    
    """
    audio_arr, sr = sf.read(input_path, samplerate=sr)
    cloud_size = sr * cloud_duration 
    if rand_seed:
        np.random.seed(rand_seed)
    grains = np.random.choice(grain_indices, n_streams) # select n random grain regions
    output_buffer = np.zeros(cloud_size)
    output_asynchronous = output_file_dir + f"/asynccloud_cloud_duration_{cloud_duration}s_{n_streams}_streams_{max_grain_density}_max_density_{sr}Hz_{rand_seed}_randseed.wav"
    for j in range(n_streams):
        main_grain = data[grains[j]] # super grain starting index
        grain_density = np.random.randint(1, max_grain_density) # once or max per second

        cloud_duration = np.max([cloud_duration,1])
        for k in range(cloud_duration):
            lower_bound = (k)*sr
            upper_bound = (k+1)*sr
            for l in range(grain_density):
                grain_duration = np.random.randint(40, 100) # 40-100 ms
                grain_size = grain_duration * 48 # in samples: 1ms is 48 samples with our sample rate
                grain_start = np.random.randint(lower_bound, upper_bound) # for the grain position
                grain_end = grain_start + grain_size # for the grain position
                if grain_end > cloud_size:
                    grain_end = cloud_size
                grain_end_audio_arr = main_grain+grain_size # the actual end index of the grain sampled from the input buffer
                grain = audio_arr[main_grain: grain_end_audio_arr]
                if len(grain) > grain_end-grain_start:
                    grain = grain[:grain_end-grain_start] 
                grain = grain*np.hanning(len(grain))
                output_buffer[grain_start: grain_end] += grain

    if np.max(np.abs(output_buffer)) > 0:
        output_buffer = output_buffer / np.max(np.abs(output_buffer))
    sf.write(output_asynchronous, output_buffer, samplerate=sr)