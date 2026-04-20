# Description thesis

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

# Background 

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

A grain is an atomic sound event, typically having a duration of 1-100ms. A grain has numerous parametres.Some general parametres prevalent in most methods of Granular Synthesis (GS) are grain size duration, grain starting point, grain waveform, stereo position, and amplitude envelope. Depending on the specific organization strategy of the grains, different parametres are most relevant. The main organization methods of granular synthesis as proposed by Curtis Roads in his book Microsound (2001) can be subdivided into two distinct branches: deterministic and non-deterministic methods. Deterministic methods of granular synthesis, such as Synchronous Granular Synthesis (SGS), Pitch-Synchronous Granular Synthesis, or Physical Models of granular synthesis, rely on deterministic global or local specifications of grain parametres. I.e. the model must either individually for each grain, or algorithmically for the set of grains, determine all the grain parametres. On the other hand, non-deterministic methods include stochastic variants, such as Quasi-SGS and Asynchronous Granular Synthesis, and, chaotic algorithms, such as a Lorentz System or a Logistic Map. Chaotic functions are deterministic in the sense that any given input will always yield the same output, however, as these systems are very sensitive to initial parametre conditions, the behaviour and output of these systems we cannot reliably predict. For this reason chaotic functions are included in the latter branch. 

Granulation is the use of a sampled sound as input for the grain waveform parametre. Instead of using a wavelet, or some other waveform, each waveform (1-100ms) is taken from this input. Due to the nature of microsounds, a lot of perceptual qualities are lost with lengths shorter than about 40 ms (Roads, 2001). 

In 

TODO: read MPEG-7 Paper

- Background on grain analysis


- What are the three algorithms, what is the previous implementation of these algorithms in music generation/ gs
    - Markov chains: transition probability matrices (Xenaxis, Miranda)
    - Logistic map

- Background on algorithms





# Methodology


# Questions: 
Q: since grain slicing/ selection is interwoven to a significant degree in a given algorithm, e.g. Markov chain of grains includes the selection. This depends on what granular parametres are handed off to the algorithm. It makes more sense to explore the workflow/ output of the three algorithms given three different types of grain analysis. Instead of having the workflow: 1) 3? grain slicing, 2) 3? grain selection, and 3) 3 grain synthesis, you would have 1) 1 grain slicing method, 2) 3 grain analysis representations: MFCC, Spectral descriptors, latent (RAVE Embedding for example), and 3) 3 algorithms that leverage the different analyses of the different representations. 