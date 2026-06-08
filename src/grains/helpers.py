import numpy as np

def rexpodec(size, decay_r=1):
    """
    Reverse exp envelope. 
    """
    if size <= 1:
        return np.ones(size)
    
    t = np.linspace(0,1,size)

    window = np.exp(-t/decay_r)

    denominator = window[0] - window[-1]
    if denominator == 0:
        return np.ones(size)
        
    normalized_window = (window - window[-1]) / denominator
    return normalized_window

def expodec(size, decay_r=1):
    """
    exp envelope. 
    """
    if size <= 1:
        return np.ones(size)
    t = np.linspace(0, 1, size)
    window = np.exp(t / decay_r)
    denominator = window[-1] - window[0]
    if denominator == 0:
        return np.ones(size)
    normalized_window = (window - window[0]) / denominator
    return normalized_window

def sinc_envelope(size, lobes=3):
    """
    Sinc envelope with n number of lobes on each side. 
    Taken from Roads Microsounds
    """
    t = np.linspace(-lobes, lobes, size)
    
    window = np.sinc(t)
    return window

def normalize_output(data):
    """
    Use max normalization for outputs to avoid clipping.
    """
    data_float = data.astype(np.float32)
    peak = np.max(np.abs(data_float))
    if peak == 0:
        normalized = data_float
    else:
        normalized = data_float / peak
    return normalized