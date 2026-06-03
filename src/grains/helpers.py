import numpy as np

def rev_exp(size, decay_r=1):
    """
    Reverse exp envelope. 
    """
    t = np.linspace(0,1,size)
    window = np.exp(-t/decay_r)
    normalized_window = (window - window[-1])/(window[0]-window[-1])
    return normalized_window

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