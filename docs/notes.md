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
    - Grain selection (Markov chain)

- Background on granular synthesis 
(Curtis Roads / )
    - Define granular parametres
    - Define different relevant methods of gs
    - Define abstract and physical models of gs
    - What are the three algorithms, what is the previous implementation of these algorithms in music generation/ gs
        - Markov chains: transition probability matrices (Xenaxis, Miranda)
        
- Background on grain analysis

- Background on algorithms



# Methodology


# Questions: 
Q: since grain slicing/ selection is interwoven to a significant degree in a given algorithm, e.g. Markov chain of grains includes the selection. This depends on what granular parametres are handed off to the algorithm. It makes more sense to explore the workflow/ output of the three algorithms given three different types of grain analysis. Instead of having the workflow: 1) 3? grain slicing, 2) 3? grain selection, and 3) 3 grain synthesis, you would have 1) 1 grain slicing method, 2) 3 grain analysis representations: MFCC, Spectral descriptors, latent (RAVE Embedding for example), and 3) 3 algorithms that leverage the different analyses of the different representations. 