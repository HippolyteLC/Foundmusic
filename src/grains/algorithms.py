import numpy as np
import soundfile as sf
from helpers import normalize_output, rev_exp

class Granulizer():
    def __init__(self, sr):
        self.sr = sr
        pass

class MarkovGranulizer(Granulizer):
    def __init__(self, sr, grain_size, densities=None, panning=[(1,1)], grain_sizes=None):
        # TODO: add second granular parametre option
        super().__init__(sr)
        self.densities = densities
        self.panning = panning 
        self.grain_sizes = grain_sizes
        self.grain_size = grain_size
        
    def rand_tpm(self, n_states, config_seed=None):
        """
        Requires own seed if run in experimental trials. 

        Returns a transition probability matrix (tpm) of 
        n_states x n_states. 

        Example usage 1) 
        If you have 3 clusters for grain waveform selection you 
        can input the n_clusters variable from the Analyzer class into
        this method to generate an according random tpm. 
        """
        if not config_seed:
            return
        
        # Part of the config random sampling 
        config_rng = np.random.default_rng(config_seed)
        random_tpm = config_rng.rand(n_states,n_states)
        tpm = []
        for i in random_tpm:
            row = i / np.sum(i)
            tpm.append(row)
        tpm = np.array(tpm)
        return tpm
    
    def run_v3(self, y, init_states, tpm, grains, dict_clusters,
                seed_grain_sampling, seed_state_sampling, 
                seed_grain_pos_sampling, n_iterations=20, delta_t=None, 
                n_streams=1, window=np.hanning, n_clusters=2):
        """
        This functions is similar to v1. For n iterations create clouds of length delta_t. Do this
        for n streams, each final stream is then added to the final output buffer. 
        Now input TPM so that I can separate seed 
        """
        if delta_t is None:
            delta_t = self.grain_size * 10
        ### param saving for metadata
        params = locals().copy()
        del params["self"]
        del params["y"]
        del params["grains"]
        del params["dict_clusters"]
        params["init_states"] = [int(i) for i in params["init_states"]]
        params["window"] = params["window"].__name__

        grain_size = int(grains[1] - grains[0])
        n_states = n_clusters * len(self.densities) * len(self.panning)
        
        # states with possible density, size values
        states = []
        clusters = list(range(n_clusters))
        for i in range(n_clusters):
            for j in range(len(self.densities)):
                for k in range(len(self.panning)):
                    states.append([
                        clusters[i], self.densities[j], self.panning[k]
                    ])
        num_chans = 1 # mono output to avoid panning influence in spectral output (not an coustic param)
        curr_states = init_states
        delta_t_samples = int(delta_t * self.sr)
        final_output_buffer = np.zeros((num_chans, n_iterations*delta_t_samples)) 

        markov_chain_tracking = []

        for stream in range(n_streams):
            output_buffer = np.array([]) 
            next_state = curr_states[stream]
            markov_chain_tracking_stream = [next_state]

            for _ in range(n_iterations):

                temp_buffer = np.zeros((num_chans, delta_t_samples)) # two output channels
                
                # maybe keep fixed grain duration? 
                cluster, density, grain_size_change = states[next_state][0], states[next_state][1] , states[next_state][2]
                
                # this sampling process needs its own seed
                state_sampling_rng = np.random.default_rng(seed_state_sampling)
                next_state = state_sampling_rng.choice(range(n_states), p=tpm[curr_states[stream]])
                curr_states[stream] = next_state
                markov_chain_tracking_stream.append(next_state)

                # this sampling process needs its own seed (not trial seed)
                grain_sampling_rng = np.random.default_rng(seed_grain_sampling)
                grain_idx = grain_sampling_rng.choice(dict_clusters[cluster]) 
                try:
                    grain_y_idx = grains[grain_idx] # grain index in the original audio array input.wav 
                except Exception as e:
                    grain_idx -= 1
                    grain_y_idx = grains[grain_idx]
                grain_y_end = grain_y_idx + grain_size
                if grain_y_end > y.shape[-1]:
                    grain_y_end = int(y.shape[-1]) 
                grain = y[grain_y_idx:grain_y_end]
                
                # add filler so that the shorter grain is sampled from the middle of the original grain
                filler = int((len(grain)-grain_size_change)//2)
                grain = grain[filler:grain_size_change+filler]
                    
                for i in range(density):
                    # this sampling process needs its own seed
                    grain_pos_sampling_rng = np.random.default_rng(seed_grain_pos_sampling)
                    s = grain_pos_sampling_rng.choice(temp_buffer.shape[-1])
                    e = s + grain_size
                    
                    if e >= temp_buffer.shape[-1]:
                        e = temp_buffer.shape[-1]
                    grain_slice = grain[:e-s]
                    grain_slice = grain_slice * window(len(grain_slice))
                    for j in range(num_chans):
                        temp_buffer[j][s:e] = temp_buffer[j][s:e] + panning[j] * grain_slice
    
                # apply screen windowing 
                # if window:
                #     for k in range(num_chans):
                #         temp_buffer[k] = temp_buffer[k] * window(temp_buffer.shape[-1])

                output_buffer = np.concatenate([output_buffer, temp_buffer], axis=1)
            final_output_buffer = normalize_output(final_output_buffer + output_buffer)
            markov_chain_tracking.append(markov_chain_tracking_stream)
        return final_output_buffer, markov_chain_tracking, params

    def run_v2(self, y, n_iterations, delta_t, n_streams, window, n_clusters, seed, init_states, grains, dict_clusters):
        """
        This functions is similar to v1. For n iterations create clouds of length delta_t. Do this
        for n streams, each final stream is then added to the final output buffer. 
        
        """
        
        params = locals().copy()
        del params["self"]
        del params["y"]
        del params["grains"]
        del params["dict_clusters"]
        params["init_states"] = [int(i) for i in params["init_states"]]
        params["window"] = params["window"].__name__

        grain_size = int(grains[1] - grains[0])
        np.random.seed(seed)
        n_states = n_clusters * len(self.densities) * len(self.panning)
        tpm = self.rand_tpm(n_states, seed=seed)
        # states with possible density, size values
        states = []
        clusters = list(range(n_clusters))
        for i in range(n_clusters):
            for j in range(len(self.densities)):
                for k in range(len(self.panning)):
                    states.append([
                        clusters[i], self.densities[j], self.panning[k]
                    ])
        num_chans = 2
        curr_states = init_states
        delta_t_samples = int(delta_t * self.sr)
        final_output_buffer = np.zeros((num_chans, n_iterations*delta_t_samples)) 

        for stream in range(n_streams):
            output_buffer = np.array([[],[]]) 
            for _ in range(n_iterations):
                temp_buffer = np.zeros((num_chans, delta_t_samples)) # two output channels
                next_state = np.random.choice(range(n_states), p=tpm[curr_states[stream]])
                curr_states[stream] = next_state
                cluster, density, panning = states[next_state][0], states[next_state][1] , states[next_state][2]
                grain_idx = np.random.choice(dict_clusters[cluster]) 
                try:
                    grain_y_idx = grains[grain_idx] # grain index in the original audio array input.wav 
                except Exception as e:
                    grain_idx -= 1
                    grain_y_idx = grains[grain_idx]
                grain_y_end = grain_y_idx + grain_size
                if grain_y_end > y.shape[-1]:
                    grain_y_end = int(y.shape[-1]) 
                grain = y[grain_y_idx:grain_y_end]
                    
                for i in range(density):
                    s = np.random.choice(temp_buffer.shape[-1])
                    e = s + grain_size
                    
                    if e >= temp_buffer.shape[-1]:
                        e = temp_buffer.shape[-1]
                    grain_slice = grain[:e-s]
                    grain_slice = grain_slice * window(len(grain_slice))
                    for j in range(num_chans):
                        temp_buffer[j][s:e] = temp_buffer[j][s:e] + panning[j] * grain_slice
    
                # apply screen windowing 
                # if window:
                #     for k in range(num_chans):
                #         temp_buffer[k] = temp_buffer[k] * window(temp_buffer.shape[-1])

                output_buffer = np.concatenate([output_buffer, temp_buffer], axis=1)
            final_output_buffer = final_output_buffer + output_buffer
        return final_output_buffer, params
    
    def run_v1(self, y, n_iterations, delta_t, n_grains, window, n_clusters, seed, init_states, grains, dict_clusters):
        """
        TBD
        n_states here will be the number clusters * number of densities
        init_states: needs to be length of the number of grains / streams
        IMPORTANT: output buffer needs to be Transposed when using save function from writing.py
        """
        params = locals().copy()
        del params["self"]
        del params["y"]
        del params["grains"]
        del params["dict_clusters"]
        params["init_states"] = [int(i) for i in params["init_states"]]
        params["window"] = params["window"].__name__

        # TODO: rethink function. Compute streams for grains separately and stack these
        # have delta t vary as a state parametre
        grain_size = grains[1] - grains[0]
        np.random.seed(seed)
        n_states = n_clusters * len(self.densities)
        tpm = self.rand_tpm(n_states, seed=seed)
        # states with possible density, size values
        states = []
        clusters = list(range(n_clusters))
        for i in range(n_clusters):
            for j in range(len(self.densities)):
                states.append([
                    clusters[i], self.densities[j]
                ])
        output_buffer = np.array([[],[]]) 
        num_chans = 2
        curr_states = init_states
        for _ in range(n_iterations):
            delta_t_samples = int(delta_t * self.sr)
            temp_buffer = np.zeros((num_chans, delta_t_samples)) # two output channels
            # multiplier = np.random.choice(range(1,7))
            # delta_t_samples = get_delta_t(multiplier)
            for i in range(n_grains):
                # get the cluster state and change the state tracking array
                next_state = np.random.choice(range(n_states), p=tpm[curr_states[i]])
                curr_states[i] = next_state
                cluster, density = states[next_state][0], states[next_state][1]

                grain_idx = np.random.choice(dict_clusters[cluster]) 
                try:
                    grain_y_idx = grains[grain_idx] # grain index in the original audio array input.wav 
                except Exception as e:
                    grain_idx -= 1
                    grain_y_idx = grains[grain_idx]

                # get the density 
                grain_y_end = grain_y_idx + grain_size
                if grain_y_end > y.shape[-1]:
                    grain_y_end = int(y.shape[-1]) # clipping if exceeds input audio bounds
                print(grain_y_end)
                grain = y[grain_y_idx:grain_y_end]
                
                # TODO: apply pitch shift?     

                if window:
                    grain = grain * window(len(grain))
            
                # get the stereo shift
                # stereo_shift = stereo_shifting[densities.index(density)]
                # print(f"grain length: {len(grain)}")
                # print(density)
                # grain_lengths.append(len(grain))
                # skip this iteration if no grain
                if density == 0:
                    continue

                # apply the density
                incr = int(temp_buffer.shape[-1] // density)
                # print("INCr:" , incr)
                for i in range(density):
                    e = i*incr + len(grain)
                    if e > temp_buffer.shape[-1]:
                        e = temp_buffer.shape[-1]
                    for j in range(num_chans):
                        temp_buffer[j][i*incr:e] = temp_buffer[j][i*incr:e] + grain[:len(temp_buffer[j][i*incr:e])]
                # apply stereo shift
                # if stereo_shift:
                #     temp_buffer[-1] = np.concatenate([np.zeros(stereo_shift), temp_buffer[-1][:-stereo_shift]])

                # apply windowing 
                if window:
                    for k in range(num_chans):
                        temp_buffer[k] = temp_buffer[k] * window(temp_buffer.shape[-1])

                output_buffer = np.concatenate([output_buffer, temp_buffer], axis=1)
            # delta_t += delta_t_incr
        # output_buffer / np.max(np.abs(output_buffer))
        return output_buffer, params


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

    