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

Synthesizer plug-ins and sample libraries are the main sources of sounds beside recording that exist to computer musicians. Recording audio in an acoustic setting is not accessible to everyone. Field recordings are recordings of audio that are not recorded in an acoustic settings; the content of a field recording can be of any sound: wind, rain, birds, a piano in a station, a phone-recorded acapella, the list goes on. Field recordings are accessible to nearly everyone. Often, the tendency is to use field recordings as  background ambient noise to define the setting or mood of a song (ref Olafur Arnalds), however, field recordings, due to their inherent lacking audio quality when compared to studio-recorded or synthetically produced sounds, are often more difficult to work with as main component of a digital music composition. There has been recent research into various ways in which input audio can be used to synthesize novel sounds. Neural latent spaces of sounds where any point in the latent space can be reconstructed into a sound is an example (irish paper/ neural GS paper/ IRCAM papers). Another example is by engel et al. that managed to output a real-time DDSP framework. However, neural methods of sound generation with sample inputs are limited in two main ways: first, outputs are biased to their training data which limits the novelty of the scope of sound generation, and second, the linkage of a sample input to synthesis parametres as in a DDSP framework loses out on much of the information that is available in a more lengthy field recording (e.g. a 1 minute recording of a metro station that is converted to parametres for a synthesis module loses out on much of the nuance of information from the original input). Methods that use the original input data in their output can avoid these two problems. T

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

A grain is an atomic sound event, typically having a duration of 1-100ms. A grain has numerous parametres.Some general parametres prevalent in most methods of Granular Synthesis (GS) are grain size duration, grain starting point, grain waveform, stereo position, and amplitude envelope. Depending on the specific organization strategy of the grains, different parametres are most relevant. The main organization methods of granular synthesis as proposed by Curtis Roads in his book Microsound (2001) can be subdivided into two distinct branches: deterministic and non-deterministic methods. Deterministic methods of granular synthesis, such as Synchronous Granular Synthesis (SGS), Pitch-Synchronous Granular Synthesis, or Physical Models of granular synthesis, rely on deterministic global or local specifications of grain parametres. I.e. the model must either individually for each grain, or algorithmically for the set of grains, determine all the grain parametres. On the other hand, non-deterministic methods include stochastic variants, such as Quasi-SGS and Asynchronous Granular Synthesis, and, chaotic algorithms, such as a Lorentz System or a Logistic Map. Chaotic functions are deterministic in the sense that any given input will always yield the same output, however, as these systems are very sensitive to initial parametre conditions, the behaviour and output of these systems we cannot reliably predict. For this reason chaotic functions are included in the latter branch. 

Granulation is the use of a sampled sound as input for the grain waveform parametre. Instead of using a wavelet, or some other waveform, each waveform (1-100ms) is taken from this input. Due to the nature of microsounds, a lot of perceptual qualities are lost with lengths shorter than about 40 ms (Roads, 2001). 

In 



- Background on grain analysis


- What are the three algorithms, what is the previous implementation of these algorithms in music generation/ gs
    - Markov chains: transition probability matrices (Xenaxis, Miranda)
    - Logistic map

- Background on algorithms


Markov theory
informatoin sou
A Markov process or chain constitutes a chain of events by which the next state or event is determined only by the current state=Markov Property. Higher order Markov chains look back a higher number of states to determine the next state. A Markov process 
Some defs: 
A state j is accessible i -> j if pn_ij > 0 for some n (where n is the number of steps). We assume every state is accessible from itself (p0: 0 steps, p0_ii = 1). Two states can communicate if they can access one another. These communicating states can be partitioned into communicating classes. A Markov chain is irreducible if all states can communicate with one another. A state is recurrent if pn_ii = 1, transient if < 1. A Markov chain is aperiodic if all its states are aperiodic. 

In discrete-time Markov chains the time spent in one state is one time unit (1) if a state has no self-transition. It is a geometric random variable otherwise, defined as geometric(1-p_ii). 
In continuous-time Markov chains the time spent in each state is a continuous random variable. 

Xenaxis Markovian Stochastic Music
1) he outlines different matrical representations of transition probabilities, denoted by two parametres for each grain parametre. I.e., for a state f0, there are two MTPs by which the transition of f0 to f1, ..., fn is determined. Which of the different MTPs is chosen depends on a grain parametre couple defined previously. 
Xenaxis mentions perturbations to the transition matrix, these perturbations ensure the sound eventually reaches an equilibrium. 
Brief explanation, he creates a coupling of parametre states to other parametre probability matrices, this way, 


# Methodology


# Questions: 
Q: since grain slicing/ selection is interwoven to a significant degree in a given algorithm, e.g. Markov chain of grains includes the selection. This depends on what granular parametres are handed off to the algorithm. It makes more sense to explore the workflow/ output of the three algorithms given three different types of grain analysis. Instead of having the workflow: 1) 3? grain slicing, 2) 3? grain selection, and 3) 3 grain synthesis, you would have 1) 1 grain slicing method, 2) 3 grain analysis representations: MFCC, Spectral descriptors, latent (RAVE Embedding for example), and 3) 3 algorithms that leverage the different analyses of the different representations. 

Q: RQ - Can markov chains propose a viable algorithmic method to granular synthesis? 

Meeting 21 april notes:
- ##

Meeting 28 april notes:
- byte tracking (RoboFlow, already implemented)