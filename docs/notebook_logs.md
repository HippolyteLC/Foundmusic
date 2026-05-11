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
##### Logs 4/5 May
Started with simple concatenative implementation sampling from the clusters unformly. This created interesting continuous sequences of grains. The transitions were determined by state transitions where states are clusters, and transition probabilities determined by the frequency of one state following another state in the original input. This created more natural sounding outputs, compared to my prior experiments in notebooks 1, 2, and 3. The number of states, or clusters, influenced how much the a certain sequence felt of a certain type, before jumping to another; the transitions were less gradual with less clusters. Musically this felt more interesting. You would imagine that more gradual jumps are typically more standard, providing for a smoother path from one sound type to another, however, the transition probabilities play a large role in this. **More research needed exploring transition probabilities for certain clusters. This will be performed in nb 5**

Wrote code that would create a joint probability matrix of two parametres, grain size and density. For the density transition matrix I used a quasi-poisson approach, where I used the poisson PMF to compute discrete probabilities, and then, normalized all rows to sum to 1 (so they remain probabilities). This turned out to not make sense for the grain size, as larger lambda values which would denote the jumps from one grain size state to another meant values not near the lambda value resulted in near 0 probabilities. 
Therefore, to keep experimenting with other parametre settings and test outputs, the grain size transition matrix was initialized randomly. 

### src\grains\5_markov_gs.ipynb
Goal: leverage sampling based on the descriptors better. Maintain transition of grain parametres with a transition matrix. 
Idea: compute the matrices with manual input. Have the grain waveform selection be linked to certain descriptor values which are linked to grain param values. E.g., grains with high centroid would be linked to low density. Maybe a work flow can then be to have the user select some points in the descriptor space, these points denote the clusters from which grains are sampled. These clusters correspond to certain descriptor regions. 

Also, try to make the outputs of this notebook more usable with each other. I.e. keep screen length and grain size parametres stable. Modulate the other parametres. 

##### Logs 6 May
Made some compositions with the simple concat approach and manually determined probabilities for the transition matrix. The simple concatenative approach has potential to be very interesting, however in longer format (longer than 10s) the lack of variation in granular parametres makes it more prone to repetition and becoming uninteresting as a standalone, perhaps the idea to introduce perturbations to the transition matrix is interesting? 
I do find that audio that isnt overwhelmingly noisy, and has distinct sound events, really does create more interesting outputs. 
When placing the composed pieces into my DAW, and adding minimal EQ'ing and some reverb, the output already has very much potential for being a standalone musical composition. An improvisation. 
**Interesting analysis: compare shorter vs longer outputs for repetitiveness and patterns in outputs produced from these simple algorithms.** 
Binary recurrence plots are an option. Function that maps to 1 if a prior frime i and a next frame j are similar below a certain value of epsilon (threshold). Can use euclidean distance, cosine distance in the librosa recurrence plot.  

Using a similar sample can result in very similar outputs if the same parametres are set. I used two different guitar recordings of mine, on the same guitar, but different voicings. The outputs with the same parametres sound very similar to the ear (TODO: provide similarity metric perhaps?). 
**How viable are markov chain models at creating novel outputs from similar inputs?** The issue here is that the grain distribution of one input of a 'guitar' will be extremely similar to the grain distribution of another 'guitar' if similar sound events happen throughout the samples. Here perhaps an idea is to find the most novel sound events, the grains farthest from any cluster of grains, and perhaps focus on synthesizing with these and comparing how similar these outputs are. 

### src\grains\6_grain_analysis.ipynb
Developing better analysis object with librosa. Computationally slower, however, better documented. 
Only contains certain methods for audio description (all documented in librosa). 
Wrote short function to show plots with two axes. 


### src\grains\7_pipeline_test.ipynb
Notebook in which the .py functions are tested, from granular analysis, to clustering, to using a dummied down version of the markov gs original nb (nb5). 
Experimented with frequencies in the sharp notes, (black piano keys) ranging from approx. 58 to 370 Hz. Interesting qualities to the output, however, obviously a tremendous amount of character of the input sound is lost, need to be kept in consideration. Otherwise, the pipeline works quite smooth, and allows for a much more rapid start to finish from inputting the sample audio, to tweaking synthesis parametres and testing the output. 
Work to be done still on the tpm distributions, as these are currently either manually, quasi-poisson, or fully stochastically initialized. Otherwise, considering other descriptor or MIR methods can still be interesting, but perhaps not adding further descriptors. Could retrieve grain pitch/ fundamental frequency, and perhaps utilize that? The grain frequency parametre could then be linked to its density, or, to its panning, volume, etc. granular parametre coupling can still be explored further. 
<!-- TODO: consider global descriptors for the grains? Such as attack, decay, etc.  -->

<!-- TODO: do a case study of the two guitar input samples and compare input and output similarities.  -->
<!-- TODO: add grain scatterplot saving for the different inputs. Compare differences between plots of different inputs. -->