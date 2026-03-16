import numpy as np
import audioflux as af
import os
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType

class FieldRecording:
    def __init__(self, path, sr):
        self.path= path
        self.sr = sr 
        
        pass
    def get_slices(self):
        """ 
        Use static or dynamic slicing to get starting 
        indices of grains. 
        """
        pass
    
    def analyze_grains(self, descriptors):
        """  
        descriptors: a list of the descriptors to be computed. I.e.
        ["centroid", "energy", "rms"]

        Compute (sub)set of all descriptors for the grains.
        Outputs metadata corresponding to these descriptors.
        
        idea: a perpetually updated csv/ json file where unique grains are 
            added with their unique descriptor values
            
        """
        pass