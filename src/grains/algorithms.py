import numpy as np
import soundfile as sf


# TODO: Add metadata saving from a render. Data might include: grain waveform, grain timestamps
# grain size, etc. for each channel of the final output audio array.

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



def logistic_map_gs(input_path, output_file_dir, data, grain_indices, 
             sr, n_iterations, r_start, r_end, x_start, 
             cloud_duration, max_grain_size, steepness, 
             rand_seed=None) -> None:
    # logistic mapping function
# x_n1 = r*x_n*(1-x_n)
    """
    If we have a time subsequence t in which we can organize some grain(s) according to the variable x_n+1 
    decided by its prior value x_n, where the range of increasing r (typically, 0-4) relates to our total
    time sequences T. 
    Can use x_n-1 to determine ratio of value in the left and right channel? 

    This function uses a logistic map where the parametre r is mapped to our iterations. In each iteration
    a buffer is created in which a single grain is positioned based on the x_n value. Other grain controls, such
    as panning, are currently determined stochastically. 
    
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
    incr = (r_end-r_start)/n_iterations
    x_n = x_start
    t_seconds = cloud_duration
    t_samples = int(t_seconds * sr)
    output_buffer = np.array([[],[]])
    output_lmgs = output_file_dir + f"output/logistic_map_gs_{n_iterations}_iter_{r_start}_r_start_{r_end}_r_end_{x_n}_x_start_{t_seconds}_t_seconds.wav"

    for _ in range(n_iterations):

        grain_buffer = np.zeros(t_samples)
        grain_idx = np.random.choice(grain_indices) # grain idx within grains_chroma array
        grain_arr_idx = data[grain_idx] # grain index within main audio array
        grain_size = np.random.randint(10, np.min([max_grain_size,t_samples]))
        grain = audio_arr[grain_arr_idx: grain_arr_idx+grain_size]
        grain_buffer_size = len(grain_buffer)
        grain_pos = int(x_n * t_samples // 1)
        grain_end = grain_pos+grain_size if not grain_size+grain_pos > grain_buffer_size else grain_buffer_size
        grain_len = len(grain) if grain_end-grain_pos > len(grain) else grain_end-grain_pos
        grain = grain[:grain_len]
        # exp window 
        lin_func = np.linspace(0, steepness, grain_len) # 5 here is arbitrary, controls the steepness
        window = 1 - np.exp(-lin_func)
        window = (window - window.min()) / (window.max() - window.min()) # normalize
        # window = np.hanning(grain_len)
        grain_buffer[grain_pos:grain_end] += grain*window

        channel = 1 if np.random.uniform() < 0.5 else 0 # uniform selection of channel assignment of amplitudes
        # print(channel)
        x_n = r_start*x_n*(1-x_n)
        x_n_1 = r_start*x_n*(1-x_n)
        r_start += incr
        l, r = x_n, x_n_1
        empty_grain_buffer = np.zeros(grain_buffer_size)

        # if channel == 1:
        #     grain_buffer = np.array([grain_buffer, empty_grain_buffer]) # maybe use vstack here instead. 
        # else:
        #     grain_buffer = np.array([empty_grain_buffer, grain_buffer])

        if channel == 1:
            # print(grain_buffer.shape)
            # print(l,r)
            new_buffer = np.array([l*grain_buffer, r*grain_buffer])
        else:
            new_buffer = np.array([r*grain_buffer, l*grain_buffer])
        # print(grain_buffer.shape)
        # print(output_buffer.shape)
        output_buffer = np.concatenate((output_buffer, new_buffer), axis=1)

    if np.max(np.abs(output_buffer)) > 0:
        output_buffer = output_buffer / np.max(np.abs(output_buffer))
    output_buffer = output_buffer.T #np.reshape(output_buffer, (output_buffer.shape[1], output_buffer.shape[0]))
    sf.write(output_lmgs, output_buffer, samplerate=sr)

    