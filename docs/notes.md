# Description thesis
Can stochastic models of granular synthesis produce viable 


# Bullet points on Introduction
IMPORTANT: why have I chosen each method. Elaborate on motivation for all the choices made. 

- Everyone has access to field recordings,
- Introduction to field recording and motivation in music production
- Introduction to signal processing and libraries
- Introducing granular synthesis as a technqiue to use sample sounds to make music
    - Corpus based granular synthesis
    - Parrticle synthesis
    - Clouds of particles and organised sotchastically/deterministic
- Introduction to algorithmic approach of producing music and also why we are doing this?
    - Chaotic system
    - Abstract models
    - Markov/ stochastic models
- Can we extract musicality, from sound not categorically recorded in a non acoustic environments, using different abstract models of granular synthesis? 

- Evaluation: 
    - Workflow for creating music from field recordings to granular analysis, to granular synthesis. 

### Introduction
Synthesizer plug-ins and sample libraries are the main sources of sounds beside recording that are used by most computer musicians. Recording high quality audio within an acoustic setting is not accessible to everyone. Field recordings are recordings of audio that are not recorded in an acoustic settings; the content of a field recording can be of any sound: wind, rain, birds, a piano in a station, a phone-recorded acapella, the list goes on. Field recordings are accessible to nearly everyone. In computer music production field recordings are often used as background ambient noise and not the centrepiece of a song (ref Olafur Arnalds). As field recordings are often longer, they also comprise various sound events. It is then more difficult to extract the totality of the sound character from a lengthy field recording in a song, when it is not simply used as background ambient noise. 
There has been recent research into various ways in which input audio, such as textural ambient sound, is encoded into a latent space, and the latent space can be decoded from, allowing for smoother and potentially novel (re)synthesis of sounds. There has also been research into using sampled sounds and large corpora of sounds for musical sound synthesis: concatenative synthesis (the concat, CataRT, Neural GS). Neural latent spaces of sounds where any point in the latent space can be reconstructed into a sound is an example (irish paper/ neural GS paper/ IRCAM papers). Another example is by engel et al. that managed to output a real-time DDSP framework. However, neural methods of sound generation with sample inputs are limited in two main ways: first, outputs are biased to their training data which limits the novelty of the scope of sound generation, and second, the linkage of a sample input to synthesis parametres as in a DDSP framework loses out on much of the information that is available in a more lengthy field recording (e.g. a 1 minute recording of a metro station that is converted to parametres for a synthesis module loses out on much of the nuance of information from the original input). Methods that use the original input data in their output avoid the abstractness problem. The novelty issue is solved dependent of the parametres of such methods. An example of such a method is granular synthesis: the arranging of microsounds, either sampled from an input sound, or granulation as in Microsound (2001, Roads), or from a generated waveform, e.g. a simple sinusoid of a certain frequency, phase, amplitude. One branch of models of granular synthesis is algorithmic modelling of the synthesis parametres. Xenaxis proposed a markovian stochastic approach to modelling grains of sounds and their parametres in a time sequence in his book Formalized Music as an attempt to provide an alternative approach to serial music composition. The stochastic algorithmic approach leverages sampling grains and their parametres from probability distributions and resolves the complexity issue of controlling parametres for many grains. Depending on the probability distributions and its parametres, such as the average rate lambda in a poisson distribution, it should take longer before a time sequence composed by such an algorithm becomes ergodic, stable. The stochastic models allow us to create novel time sequences of grains. 
This paper proposes an approach in python to algorithmic music which leverages the characteristics of field recordings to be musical centrepieces. 

RQ: Are markov chains implementrd

# Background 

TODO (week april 19th): read MPEG-7 Paper, read Caterpillar paper, add notes, read through fundamentals of Markov Processes/ Markov transition matrix/ Markov Chains >> Read Markov generative music papers (or GS paper) / implement different modes of classifying grains, i.e. asserting grain states to transition to- and from. Also, consider if another GS method might be more suited for exploring interesting temporal structures in combi. Check datasets used in Neurogranular synthesis paper, The Concatenator / Let it Bee papers (or other papers in the field). Delve into math of spectral descriptors

- Background on digital signal processing
    - Frequency / Time representations of sound
    - Spectral analysis of sound
    - Explain spectral descriptors (refer to AudioFlux) library
    - Spectral descriptors for grains (CataRT)
    - Slicing / Onset detection / other

- Background on granular synthesis 
(Curtis Roads / )
    - Define granular parametres
    - Define different relevant methods of gs
    - Define abstract and physical models of gs

### DSP Acoustic descriptors
To Cite: Peeters et al (2001), Klapuri et al (DSP for msuic transcription, 2006)
Note: X(k) refers to a vector of RMS levels of sub bands or a DFT spectrum at a time t (a certain frame)
##### Spectral Features
- Spectral flatness
![Spectral Flatness Formula](\images\spectral_flatness_formula_klapuri_et_al.png)
Describes how flat the spectrum of a sound is. The flatter the sound, a high value, is a noisy sound whereas the peakier the sound, a low value, is a tonal sound. Ratio of geometric mean to arithmetic mean of an analysis frame. 
- Spectral Flux
Frame by frame spectral change by comparing the distance between two spectrums at t and t-1.
- Spectral Rolloff
Denotes how much of the frequenc
- Zero Crossing Rate
![Zero Crossing Rate Formula](\images\zcr_formula_klapuri_et_al.png)
Is describecfd as the number of times a signal changes sign. It is strongly correlated to the spectral centroid. It describes how much high-frequency content a signal contains. It is effective at discriminating different classes of percussive instruments.

##### Temporal features 
Temporal features are not used in the grain analysis. Temporal features are often used to describe things about the amplitude envelope of signals. However, since the grains sampled in this paper are rather short (100ms) and grains are arbitrarily indexed, temporal descriptors are less valuable. E.g. if we break a sound event by taking an arbitrary micro acoustic event within it, we lose the valuable informatoin that an amplitude envelope tells us about the sound event. 

### Granular Synthesis
A grain is an atomic sound event, typically having a duration of 1-100ms. A grain has numerous parametres.Some general parametres prevalent in most methods of Granular Synthesis (GS) are grain size duration, grain starting point, grain waveform, stereo position, and amplitude envelope. Depending on the specific organization strategy of the grains, different parametres are most relevant. The main organization methods of granular synthesis as proposed by Curtis Roads in his book Microsound (2001) can be subdivided into two distinct branches: deterministic and non-deterministic methods. Deterministic methods of granular synthesis, such as Synchronous Granular Synthesis (SGS), Pitch-Synchronous Granular Synthesis, or Physical Models of granular synthesis, rely on deterministic global or local specifications of grain parametres. I.e. the model must either individually for each grain, or algorithmically for the set of grains, determine all the grain parametres. On the other hand, non-deterministic methods include stochastic variants, such as Quasi-SGS and Asynchronous Granular Synthesis, and, chaotic algorithms, such as a Lorentz System or a Logistic Map. Chaotic functions are deterministic in the sense that any given input will always yield the same output, however, as these systems are very sensitive to initial parametre conditions, the behaviour and output of these systems we cannot reliably predict. For this reason chaotic functions are included in the latter branch. 

Granulation is the use of a sampled sound as input for the grain waveform parametre. Instead of using a wavelet, or some other waveform, each waveform (1-100ms) is taken from this input. Due to the nature of microsounds, a lot of perceptual qualities are lost with lengths shorter than about 40 ms (Roads, 2001). 


- Background on grain analysis


- What are the three algorithms, what is the previous implementation of these algorithms in music generation/ gs
    - Markov chains: transition probability matrices (Xenaxis, Miranda)
    - Logistic map

- Background on algorithms


### Markov theory
informatoin sou
A Markov process or chain constitutes a chain of events by which the next state or event is determined only by the current state=Markov Property. Higher order Markov chains look back a higher number of states to determine the next state. A Markov process 
Some defs: 
A state j is accessible i -> j if pn_ij > 0 for some n (where n is the number of steps). We assume every state is accessible from itself (p0: 0 steps, p0_ii = 1). Two states can communicate if they can access one another. These communicating states can be partitioned into communicating classes. A Markov chain is irreducible if all states can communicate with one another. A state is recurrent if pn_ii = 1, transient if < 1. A Markov chain is aperiodic if all its states are aperiodic. 

In discrete-time Markov chains the time spent in one state is one time unit (1) if a state has no self-transition. It is a geometric random variable otherwise, defined as geometric(1-p_ii). 
In continuous-time Markov chains the time spent in each state is a continuous random variable. 

- Xenaxis Markovian Stochastic Music
he outlines different matrical representations of transition probabilities, denoted by two parametres for each grain parametre. I.e., for a state f0, there are two MTPs by which the transition of f0 to f1, ..., fn is determined. Which of the different MTPs is chosen depends on a grain parametre couple defined previously. 
Xenaxis mentions perturbations to the transition matrix, these perturbations ensure the sound eventually reaches an equilibrium. 
Brief explanation, he creates a coupling of parametre states to other parametre probability matrices, this way, 


# Methodology

- Design science: 
    - Keep track and log production process
    - Note moments of creative production - link to some output
    - Document failures

### Grain analysis
<!-- TODO: Cite Fundamentals of music processing here : Meinard Muller-->
The original input .wav file is loaded into the Analyzer class. Time-varying descriptors are then computed using the Librosa spectral features. For fast computation of the Short-Time Fourier Transform (STFT), the size of the window, the number of bins, and the hop size all need to be powers of two; this is a feature of the Fast Fourier Transform (FFT). The window size is set to 2048 samples. Considering our sample rate of 48kHz, the lowest detectable frequency becomes ≈23.4Hz. This is right about the lower hearing bound of human hearing (≈20Hz). In signal analysis, the time resolution is inversely proportional to the frequency resolution. If we increase our window size, we capture less time snapshots of the evolution of a signal, thus resulting in a lower temporal frequency. With the sampling rate at 48kHz, the window duration is $2048/48000≈0.043$. The input signal is split into windows of 43ms. If we want to increase our time resolution, we must take one power of two less in our window size, thus, 1024 samples, and so forth. The frequency resolution is determined by the numbers of frequency bins we consider in our window. By the Nyquist frequency we only need frequency components below half of our sampling rate. In this case we need only consider frequencies below 24kHz. Typically, our FFT has the same size as our window, thus, 2048. These can be adjusted to increased time or frequency resolution. The size of our fft, i.e. the number of samples we consider, is thus both linked directly to the duration of our window and the number of frequency bins we can consider. 
The hop size is set to 512 and determines the distance between one column computed of a signal and the next. The window uses a hanning window to ensure smooth windowing. To ensure all information of a signal is captured in the STFT, the hop length is typically set to $window length//4$, therefore 512. From the STFT we can now compute any of the available descriptors: spectral centroid, rolloff, contrast, rms, and zero-crossing rate. 

To associate descriptor values with the grains, the stft frames over which the descriptor values are computed are reformatted to fit the sample size of the original input ($total sample size = sample rate * input duration$). There is now a descriptor value at each sample point, which allows us to take the mean and standard deviation of each sample window corresponding to our grain size. Each grain now has an adhering mean/ std of some descriptor value. As our analysis window is approximately 43ms, and our grains are 100ms, each grain takes a mean of ≈2.3 analysis window values. 

We have now managed to compute descriptors for each grain. The current problem is that we want to analyze grains from different inputs. An optimal approach might be 

- PCA for dimensionality reduction or visualization
- k-means
- gaussian mixture models
- DBSCAN 




<!-- IDEA: instead of using only two descriptors, which leave a lot to be desired in terms of acoustic description capability, can instead use a vector of selected features, and use PCA + CDA to reduce dimensionality. Then, the clusters can serve as points to potentially sample from? Klapuri et al, recommend using 10x fewer features than that you have training instances. So then,  -->

### Discussion

- Dynamic analysis of which descriptors for a given input. I.e. which descriptor vector best discriminates between all the different sound events present in the input. Again, for which set of descriptors can similar grains have low variance, and different grains have high variance.


# Questions 
Q: since grain slicing/ selection is interwoven to a significant degree in a given algorithm, e.g. Markov chain of grains includes the selection. This depends on what granular parametres are handed off to the algorithm. It makes more sense to explore the workflow/ output of the three algorithms given three different types of grain analysis. Instead of having the workflow: 1) 3? grain slicing, 2) 3? grain selection, and 3) 3 grain synthesis, you would have 1) 1 grain slicing method, 2) 3 grain analysis representations: MFCC, Spectral descriptors, latent (RAVE Embedding for example), and 3) 3 algorithms that leverage the different analyses of the different representations. 

Q: RQ - Can markov chains propose a viable algorithmic method to granular synthesis? 

Meeting 21 april notes:

Meeting 28 april notes:
- byte tracking (RoboFlow, already implemented)

# Ideas
1. use descriptor values in grain parametre setting/ probability distribution coupling.
2. stat analysis between different hop window sizes 