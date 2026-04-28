### src\grains\3_stochastic_gs.ipynb
- Extracted slice indexes from "corpus\metro_sample_2\input.wav" 
- Computing chroma_cqts (Constant-Q Transform) for each grain, to extract the chromatic note of each grain
- Implemented asynchronous granular synthesis with randomized parametre selection
- Implemented granular synthesis with logistic map for grain start position selection: this algorithms also includes some stochastic grain parameterization

### src\grains\2_grain_analysis.ipynb
- Clustered predetermined grains based on all descriptors in grains.json
- Linear interpolation from a given starting point to and end point in the descriptor space is used to compute the n nearest neighbours along this trajectory. These are concatenated into an output audio array

### src\grains\4_markov_gs.ipynb
- Fixed length slicing, analysis of the whole input sound, hop_size corresponding to grain_size
- Clustering kmeans based on rms/ centroid distance
- Transition probability matrix used for clusters, then uniform random sampling from each cluster
- Added Xenaxis inspired screen method. Each screen contains a max number of grains, each of these grains has its own transition matrices. The first for the grain waveform selection, the second, for the grain density and size parametres. Using a multinomial PMF using a quasi-poisson approach. 
- Added parametre saving and hashing of outputs (wav) and outputs (json). 

