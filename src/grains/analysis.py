import numpy as np
import os
import subprocess
import audioflux as af
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType



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

