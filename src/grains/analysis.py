import numpy as np
import os
import audioflux as af
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType
import matplotlib.pyplot as plt
import sklearn
import librosa

def get_spectrogram(data, y_axis, x_axis, sr, title=None):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=D,y_axis=y_axis, x_axis=x_axis, sr=sr, ax=ax)
    ax.set(title='Linear-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

class AnalysisObject():
    def __init__(self, dir, sr):
        self.dir = dir
        self.sr = sr
        self.input = os.path.normpath(self.dir + "\\input.wav")
        self.data = None

    def exists(self):
        return os.path.exists(self.input)

    def load_soundfile(self):
        print(self.input)
        if os.path.exists(self.input):
            y, _ = af.read(path=self.input, samplate=self.sr)
            self.data = y
    
    def define_grains(self, grain_duration):
        """
        compute grain duration as s x samples operation
        return grain starting indexes in the original audio array (self.y)
        """
        grain_size = int(grain_duration * self.sr)
        if not self.len_input:
            _, _ = self.load_soundfile()
        n_grains_in_source = int(len(self.data) // grain_size)
        grains = [i*grain_size for i in range(n_grains_in_source)]
        return grains
    
    def get_spectral_arr(self, grain_duration, num_freq_bins=None, radix_exp=None):
        """
        Each frame corresponds to a grain due to slide_length=grain_size. 
        The other parametres are set to default analysis values. 
        Alternative recommended values to be considered in the time/ frequency 
        accuracy trade-off: 4096, 2048, 1024 (then follows for num_freq_bins: 2049, 1025, 513, etc.)
        Radix_exp is 2^12: 4096, increase or decrease accordingly with num_freq_bins. 
        returns a spec arr of shape (2049, n_grains-1), this is computationally useful.
        We only compute the entire audio's BFT and spectral arr once. More lacking in accuracy though (resolution). 
        """
        grain_size = int(grain_duration * self.sr)

        if not num_freq_bins:
            num_freq_bins = 2049 # standard FFT values
        if not radix_exp:
            radix_exp = 12 #2^12
        bft_obj = af.BFT(num=num_freq_bins, samplate=self.sr, radix2_exp=radix_exp, slide_length=grain_size,
            data_type=af.type.SpectralDataType.MAG,
            scale_type=af.type.SpectralFilterBankScaleType.LINEAR)       
        spec_arr = bft_obj.bft(self.y)
        spec_arr = np.abs(spec_arr)
        spectral_obj = af.Spectral(num=bft_obj.num,
                                fre_band_arr=bft_obj.get_fre_band_arr())
        n_time = spec_arr.shape[-1]  
        spectral_obj.set_time_length(n_time)

        return spec_arr, spectral_obj #return spec_arr and obj to compute descriptors

    def get_descriptor_array(self, descriptor, grain_duration):
        """
        descriptor: one of the following methods [centroid, rms, crest]
        """
        spec_arr, spectral_obj = self.get_spectral_arr(grain_duration)
        return spectral_obj.descriptor(spec_arr)
    
    def get_cluster_obj(self, n_clusters, arr_1, arr_2):
        """ 
        here n_init defaults to 1, the number of runs with diff centroid seeds. 
        random_state: use int to make deterministic
        """
        x = np.array([[i, j] for i, j in zip(arr_1, arr_2)])
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(x) 
        return kmeans, arr_1, arr_2

    def display_scatter_plot(self, arr_1, arr_2):
        plt.figure(figsize=(10, 7))
        plt.scatter(arr_1, arr_2, alpha=0.7)
        # plt.ylabel()
        # plt.xlabel("Spectral Root Mean Square")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def get_dict_clusters(self, n_clusters, arr_1, arr_2):
        kmeans, _, _ = self.get_cluster_obj(self, n_clusters, arr_1, arr_2)
        dict_clusters = {}
        for idx, lab in enumerate(kmeans.labels_):
            dict_clusters[lab] = dict_clusters.get(lab, [])
            dict_clusters[lab].append(idx)
        return dict_clusters

dir = "..\..\corpus\metro_sample_1"

sr = 48000
obj = AnalysisObject(dir,sr=sr)
obj.dir
y, sr = librosa.load(path=obj.input, sr=obj.sr)
print(obj.data)
obj.load_soundfile()
print(obj.exists())


def compute_spectral_descriptors(file_path, sr, output_dir, grain_size):
    """
    grain_size: in samples (consider the sample rate)
    """
    y, sr = af.read(file_path)
    audio_len = len(y)
    slice_indices = np.array(
        [i*grain_size for i in range(audio_len//grain_size)]
    )
    
    if y.ndim > 1:
        y=np.mean(y, axis=1)
        
    grains_descriptors = []
    for index in slice_indices:
        grain_end = index+grain_size # 1ms to 100ms
        if grain_end > audio_len:
            grain_end = audio_len
        s, e = index, grain_end
        grain = y[s: e]
        # print(grain)
        bft_obj = af.BFT(num=64, samplate=sr, radix2_exp=7,
                    data_type=SpectralDataType.MAG,
                    scale_type=SpectralFilterBankScaleType.LINEAR)
        spec_arr = bft_obj.bft(grain)
        spec_arr = np.abs(spec_arr)
        
        # Create Spectral object and extract spectral feature
        spectral_obj = af.Spectral(num=bft_obj.num,
                                fre_band_arr=bft_obj.get_fre_band_arr())
        spectral_obj.set_time_length(spec_arr.shape[-1])
        
        d ={
            "source_id": 0,
            "grain_start": int(s),
            "grain_size": len(grain),
            "centroid": float(np.mean(spectral_obj.centroid(spec_arr))),
            # "flatness": float(np.mean(spectral_obj.flatness(spec_arr))),
            # "kurtosis": float(np.mean(spectral_obj.kurtosis(spec_arr))),
            "flux": float(np.mean(spectral_obj.flux(spec_arr))),
            "energy": float(np.mean(spectral_obj.energy(spec_arr))),
            # "crest": float(np.mean(spectral_obj.crest(spec_arr))),
            "rms": float(np.mean(spectral_obj.rms(spec_arr))),
            # "eef": float(np.mean(spectral_obj.eef(spec_arr))),
            # "eer": float(np.mean(spectral_obj.eer(spec_arr))),
            # "band_width": float(np.mean(spectral_obj.band_width(spec_arr))),
            # "decrease": float(np.mean(spectral_obj.decrease(spec_arr))),
            "entropy": float(np.mean(spectral_obj.entropy(spec_arr))),
            "spread": float(np.mean(spectral_obj.spread(spec_arr))),
            # "slope": float(np.mean(spectral_obj.slope(spec_arr)))
        }
        grains_descriptors.append(d)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, f"{file_path[7:-4]}.json")
    with open(path, "w") as f:
        # json.dump(grains_descriptors, f, indent=4)
        pass
    # print(f"Saved {len(df)} grains to {output_parquet}")
