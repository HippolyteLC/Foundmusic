import numpy as np
import os
import audioflux as af
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType
import matplotlib.pyplot as plt
import sklearn
import librosa
from writing import get_parametre_hashing
import json 
import csv
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler



class Analyzer():
    """
    Analysis object with Librosa. Loading with af. All descriptor methods utilize STFT representation
    for faster computation it is provided globally. 
    5 descriptor methods are added, some of which pertain to different descriptor categories as 
    per Peeters' paper. 
    Using specific 2048 n_fft and window length, and 512 hop length for good freq/ time resolution trade off
    Conversion of these window values to the grain descriptor values is included
    """

    def __init__(self, path, sr):
        self.path = path
        self.sr=sr
        self.input_path = os.path.normpath(path + "\\input.wav")
        self.y = None
        self.stft = None
        self.loaded_y=False
        self.loaded_stft=False

    def compute_stft(self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length = None,
        window = "hann",
        center: bool = True,
        pad_mode = "constant",
        ):
        if not self.loaded_y:
            self.load_audio_data()
            self.loaded_y=True
        stft = librosa.stft(
            y=self.y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        self.stft = np.abs(stft)

    def load_audio_data(self):
        y, _ = af.read(path=self.input_path, samplate=self.sr)
        self.y = y
    
    def compute_zrc(self, n_fft=2048, hop_length=512):
        """ 
        Compute ZRC.
        """
        if not self.loaded_stft:
            self.compute_stft()
            self.loaded_stft=True
        arr = librosa.feature.zero_crossing_rate(S=self.stft, n_fft=n_fft, hop_length=hop_length)
        return arr[0] 

    def compute_flatness(self, n_fft=2048, hop_length=512):
        """ 
        Compute spectral Flatness, standard Hanning window. 
        """
        if not self.loaded_stft:
            self.compute_stft()
            self.loaded_stft=True
        arr = librosa.feature.spectral_flatness(S=self.stft, n_fft=n_fft, hop_length=hop_length)
        return arr[0]
    
    def compute_rms(self, frame_length=2048, hop_length=512):
        """ 
        We use the Audio data 
        """
        if not self.loaded_stft:
            self.compute_stft()
            self.loaded_stft=True
        arr = librosa.feature.rms(S=self.stft, frame_length=frame_length, hop_length=hop_length)
        return arr[0]
    
    def compute_centroid(self, n_fft=2048, hop_length=512):
        """ 
        We use the Audio data 
        """
        if not self.loaded_stft:
            self.compute_stft()
            self.loaded_stft=True
        arr = librosa.feature.spectral_centroid(S=self.stft, n_fft=n_fft, hop_length=hop_length)
        return arr[0]
    
    def compute_rolloff(self, n_fft=2048, hop_length=512):
        """ 
        We use the Audio data 
        """
        if not self.loaded_stft:
            self.compute_stft()
            self.loaded_stft=True
        arr = librosa.feature.spectral_rolloff(S=self.stft, n_fft=n_fft, hop_length=hop_length)
        return arr[0]
    
    def convert_descriptor_arr(self, descriptor_arr):
        """ 
        Grain duration in seconds. 
        Convert the descriptor arr to have descr values for each sample, can then compute per grain
        descr value. 
        """
        if self.loaded_y:
            incr = self.y.shape[-1]//descriptor_arr.shape[-1]
            buffer = []
            index = -1
            for i in range(self.y.shape[-1]):    
                if i % incr == 0:
                    index+=1
                if index == descriptor_arr.shape[-1]:
                    break
                buffer.append(descriptor_arr[index])
            difference = self.y.shape[-1] - len(buffer)
            if difference > 0: # padding the buffer 
                buffer.extend([descriptor_arr[-1] for _ in range(difference)])
            return buffer 
        
    def compute_all_descriptors(self, grain_duration):
        grain_size = int(self.sr*grain_duration)
        n_grains = int(len(self.y)//grain_size)
        grains = [i*grain_size for i in range(n_grains)]

        zrc_arr = self.compute_zrc()
        flatness_arr = self.compute_flatness()
        rms_arr = self.compute_rms()
        rolloff_arr = self.compute_rolloff()

        zrc_array = self.convert_descriptor_arr(zrc_arr)
        zrc_grain_descr, _, _ = self.get_grain_descriptors(grain_size=grain_size, descriptor_arr=zrc_array)
 
        flatness_array = self.convert_descriptor_arr(flatness_arr)
        flatness_grain_descr, _, _ = self.get_grain_descriptors(grain_size=grain_size, descriptor_arr=flatness_array)

        rms_array = self.convert_descriptor_arr(rms_arr)
        rms_grain_descr, _, _ = self.get_grain_descriptors(grain_size=grain_size, descriptor_arr=rms_array)

        rolloff_array = self.convert_descriptor_arr(rolloff_arr)
        rolloff_grain_descr, _, _ = self.get_grain_descriptors(grain_size=grain_size, descriptor_arr=rolloff_array)

               
    def get_grain_descriptors(self, grain_size, descriptor_arr):
        print(f"Length y: {len(self.y)} Length descr y: {len(descriptor_arr)}")
        grain_mean_descr = []
        grain_std_descr = []
        n_grains = int(len(descriptor_arr)//grain_size)
        for i in range(n_grains):
            grain_dscr_mean = np.mean(descriptor_arr[i*grain_size:(i+1)*grain_size])
            grain_dscr_std = np.std(descriptor_arr[i*grain_size:(i+1)*grain_size])
            grain_mean_descr.append(grain_dscr_mean)
            grain_std_descr.append(grain_dscr_std)
        return grain_mean_descr, grain_std_descr, descriptor_arr
    
    def n_grains(self, grain_duration):
        grain_size = int(self.sr*grain_duration)
        if not self.loaded_y:
            self.load_audio_data()
        return int(len(self.y)//grain_size)

def show_scatter_plt(x, y, x_label, y_label, title, alpha=0.7):
    """ 
    Show scatter plot for two arrays. Add title, x and y labels. 
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=alpha)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=alpha)
    plt.show()
    

def get_spectrogram(data, y_axis, x_axis, sr, title=None):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=D,y_axis=y_axis, x_axis=x_axis, sr=sr, ax=ax)
    ax.set(title='Linear-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")



### Class using AudioFlux below
# --------------------------------------------------------------------------#

class AnalyzerObject():
    """
    Using specific 2048 n_fft and window length, and 512 hop length for good freq/ time resolution trade off
    Conversion of these window values to the grain descriptor values is included
    """

    def __init__(self, path, sr):
        self.path = path
        self.sr=sr
        self.input_path = os.path.normpath(path + "\\input.wav")
        self.metadata = os.path.normpath(path + "\\metadata")
        self.y = None
        self.stft = None
        self.loaded_y=False
        self.loaded_stft=False

    def load_y(self):
        y, _ = af.read(path=self.input_path, samplate=self.sr)
        self.y = y
        self.loaded_y = True
    def get_spectral_arr(self, num_freq_bins=1025, radix_exp=11):
        """
        Each frame corresponds to a grain due to slide_length=grain_size. 
        The other parametres are set to default analysis values. 
        Alternative recommended values to be considered in the time/ frequency 
        accuracy trade-off: 4096, 2048, 1024 (then follows for num_freq_bins: 2049, 1025, 513, etc.)
        Radix_exp is 2^11: 2048, increase or decrease accordingly with num_freq_bins. 
        returns a spec arr of shape (2049, n_grains-1), this is computationally useful.
        Slide length is automatically set to fft_window size / 4, so in this case 2^11 = 2048, 2048/4, 512. 
        We only compute the entire audio's BFT and spectral arr once. 
        """
        if not self.loaded_y:
            self.load_y()
        bft_obj = af.BFT(num=num_freq_bins, samplate=self.sr, radix2_exp=radix_exp, 
            data_type=af.type.SpectralDataType.MAG,
            scale_type=af.type.SpectralFilterBankScaleType.LINEAR)       
        spec_arr = bft_obj.bft(self.y)
        spec_arr = np.abs(spec_arr)
        spectral_obj = af.Spectral(num=bft_obj.num,
                                fre_band_arr=bft_obj.get_fre_band_arr())
        n_time = spec_arr.shape[-1]  
        spectral_obj.set_time_length(n_time)
        return spec_arr, spectral_obj 

    def convert_descriptor_arr(self, descriptor_arr):
        """ 
        Grain duration in seconds. 
        Convert the descriptor arr to have descr values for each sample, can then compute per grain
        descr value. 
        """
        incr = self.y.shape[-1]//descriptor_arr.shape[-1]
        buffer = []
        index = -1
        for i in range(self.y.shape[-1]):    
            if i % incr == 0:
                index+=1
            if index == descriptor_arr.shape[-1]:
                break
            buffer.append(descriptor_arr[index])
        difference = self.y.shape[-1] - len(buffer)
        if difference > 0: # padding the buffer 
            buffer.extend([descriptor_arr[-1] for _ in range(difference)])
        return buffer 
    
    def get_grain_descriptors(self, grain_size, descriptor_arr):
        """
        Compute mean of per sample descriptor value for a grain of a certain size
        from the full descriptor array from original input. 
        Input descriptor arr contains per sample values.
        """
        print(f"Length y: {len(self.y)} Length descr y: {len(descriptor_arr)}")
        grain_mean_descr = []
        n_grains = int(len(descriptor_arr)//grain_size)
        for i in range(n_grains):
            grain_dscr_mean = np.mean(descriptor_arr[i*grain_size:(i+1)*grain_size])
            grain_mean_descr.append(grain_dscr_mean)
        return grain_mean_descr
    
    def compute_grain_descriptors(self, grain_duration, num_freq_bins=1025, radix_exp=11):
        """
        Compute centroid, spread, skewness, kurtosis (good for percussive
        discrimination in instrument classification, Ansi Klapuri et al. 2006)
        Compute flux, rolloff, flatness (good for instrument discrimination, Klapuri et al. 2006)
        list of dict to df of per grain features.
        df shape: (n_samples, n_features)
        """
        spec_arr, spectral_obj = self.get_spectral_arr(num_freq_bins, radix_exp)
        grain_size = int(self.sr*grain_duration)
        n_grains = int(len(self.y)//grain_size)
        grains = [i*grain_size for i in range(n_grains)]
        centroid = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.centroid(spec_arr)))
        flux = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.flux(spec_arr)))
        rolloff = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.rolloff(spec_arr)))
        flatness = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.flatness(spec_arr)))
        spread = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.spread(spec_arr)))
        skewness = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.skewness(spec_arr)))
        kurtosis = self.get_grain_descriptors(grain_size, self.convert_descriptor_arr(spectral_obj.kurtosis(spec_arr)))

        grain_metadata = []
        for i in range(len(grains)):
            grain_descriptors = {}
            grain_descriptors["index"] = grains[i]
            grain_descriptors["sr"] = self.sr
            grain_descriptors["size"] = grain_size
            grain_descriptors["centroid"] = centroid[i]
            grain_descriptors["flux"] = flux[i]
            grain_descriptors["rolloff"] = rolloff[i]
            grain_descriptors["flatness"] = flatness[i]
            grain_descriptors["spread"] = spread[i]
            grain_descriptors["skewness"] = skewness[i]
            grain_descriptors["kurtosis"] = kurtosis[i]
            grain_descriptors["id"] = get_parametre_hashing(grain_descriptors, hash_length=9)
            grain_metadata.append(grain_descriptors)
        # file_params = {"grain_size": grain_size, "descriptors": list(grain_descriptors.keys())}
        df = pd.DataFrame(grain_metadata)
        # TODO: instead change this to use df.loc for selecting only descriptor columns in the data scaling.
        # indexing_l = [-1, 0, 2, 1].extend(list(range(3, df.shape[-1])))
        df = df[["id", "sr", "index", "size", "centroid", "flux", "rolloff", "flatness", "spread", "skewness", "kurtosis"]] 
        return df

    def save_metadata(self, df):
        """
        Save df to csv file of per grain descriptors.
        """
        file_params = df.iloc[-1].to_dict()
        file_name = get_parametre_hashing(file_params, hash_length=8)
        file_path = os.path.normpath(self.metadata + "\\grain_metadata_" + str(file_name) + ".csv")
        df.to_csv(file_path, index=False)
        print(f"Saved to csv to: {file_path}")

    def load_metadata(self, path):
        """
        load csv as pandas df
        """
        df = pd.read_csv(path)
        return df
    
    def scale_metadata(self, path):
        # TODO: change input, currently does not make sense
        """
        Use StandardScalar class from sklearn to scale data. 
        Important for unsupervised algorithms (GMMs and KMeans) or for
        feature projection algs (PCA, HDBSCAN, etc.)
        """
        df = self.load_metadata(path)
        df_scaled = df
        scaler = StandardScaler()
        df_descriptors_to_scale = df_scaled[["centroid", "flux", "rolloff", "flatness", "spread", "skewness", "kurtosis"]]
        df_scaled[["centroid", "flux", "rolloff", "flatness", "spread", "skewness", "kurtosis"]] = scaler.fit_transform(df_descriptors_to_scale)
        return df, df_scaled # return original and scaled df 
    
    def compute_kmeans(self, path, n_clusters, features=None):
        """
        KMeans algorithm on scaled per grain feature data
        features: a list of strings representing desired analysis features
        returns kmeans object
        """
        _, df_scaled = self.scale_metadata(path)
        if not features:
            features = ["centroid", "flux", "rolloff", "flatness", "spread", "skewness", "kurtosis"]
            features_scaled = df_scaled[features] # use the columns corresponding to the grain descriptors
        else:
            features_scaled = df_scaled[features]
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=1, random_state=0).fit(features_scaled)
        return kmeans # return kmeans object 
    
    def get_cluster_dict(self, path, n_clusters, features=None):
        """
        computes kmeans object and writes data to a dictionary. 
        Useful for granular synthesis algorithms that utilize cluster based
        grain sampling.
        """
        kmeans = self.compute_kmeans(path, n_clusters, features)
        dict_clusters = {}
        for idx, lab in enumerate(kmeans.labels_):
            dict_clusters[lab] = dict_clusters.get(lab, [])
            dict_clusters[lab].append(idx)
        return dict_clusters, kmeans
    
    def grains(self, grain_duration):
        if not self.loaded_y:
            self.load_y()
        grain_size = int(self.sr*grain_duration)
        n_grains = int(len(self.y)//grain_size)
        grains = [i*grain_size for i in range(n_grains)]
        return grains



### OBSOLETE FUNCTIONS

    # def get_cluster_obj(self, n_clusters, arr_1, arr_2):
    #     """ 
    #     here n_init defaults to 1, the number of runs with diff centroid seeds. 
    #     random_state: use int to make deterministic
    #     """
    #     x = np.array([[i, j] for i, j in zip(arr_1, arr_2)])
    #     kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(x) 
    #     return kmeans, arr_1, arr_2

    # def display_scatter_plot(self, arr_1, arr_2):
    #     plt.figure(figsize=(10, 7))
    #     plt.scatter(arr_1, arr_2, alpha=0.7)
    #     plt.ylabel("Spectral Flatness")
    #     plt.xlabel("Spectral Root Mean Square")
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.show()
    
    # def save_scatter_plot(self, arr_1, arr_2):
    #     y_label = "Spectral Flatness"
    #     x_label = "Spectral Root Mean Square"
    #     plt.figure(figsize=(10, 7))
    #     plt.scatter(arr_1, arr_2, alpha=0.7)
    #     plt.ylabel(y_label)
    #     plt.xlabel(x_label)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     save_title = f""
    #     plt.savefig()

    # def get_dict_clusters(self, n_clusters, arr_1, arr_2):
    #     kmeans, _, _ = self.get_cluster_obj(self, n_clusters, arr_1, arr_2)
    #     dict_clusters = {}
    #     for idx, lab in enumerate(kmeans.labels_):
    #         dict_clusters[lab] = dict_clusters.get(lab, [])
    #         dict_clusters[lab].append(idx)
    #     return dict_clusters




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
