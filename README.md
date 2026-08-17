# classical-guitar-string-fret-localisation
The same note can be played in different string/fret positions on guitar. For example, E4 (330Hz) can be played with the open high E string (“string 1 fret 0”), string 2 fret 5 or string 3 fret 9. Although the note is the same, it sounds different in tone and colour.
The aim of this project is to distinguish the string/fret position that a particular note was played in based on the audio recording.

The minimum viable input is a recording of a single isolated plucked note of a classical guitar in standard tuning.
Frets 0 to 12 will be considered on each string. This spans one octave on each string and captures the main alternative string/fret positions while keeping the number of positions and required recordings tractable for a proof-of-concept. Higher frets could be incorporated using the same pipeline in future work.

The minimum viable output should be the string and fret position. In future development, factors like inference time and classification confidence may also be explored.


# aspects to consider
Timbre is the unique quality or “colour” of sound. Words often used to describe timbre are “bright”, “warm”, “mellow”, “dark” or “piercing. Relevant contributors to timbre include the spectral distribution of harmonics, the physical properties of the vibrating system, and temporal characteristics such as attack and decay.

When a note is plucked, there are extra vibrations at higher frequencies than the note being played. Some strings may produce stronger higher harmonics when they are plucked, making them sound brighter or sharper. Others may have fewer harmonics, resulting in them sounding warmer/more mellow.
Considering the ratios of different harmonics may help reveal whether the sound is brighter or warmer e.g. a higher hm/hn amplitude ratio (where m > n) indicates greater relative energy in the higher harmonic and may therefore be associated with a brighter timbre.

A spectral centroid could also be useful. This feature measures the brightness of a sound by calculating a weighted mean of the frequencies present in a signal, using their magnitudes as weights:
S=(∑_(k=0)^(N-1)▒f (k)⋅X(k))/(∑_(k=0)^(N-1)▒X (k) )
Where N is the total number of frequency bins in the frame, X is the amplitude of the signal at frequency bin k, and f is the frequency corresponding to bin k.
The frequency of any given bin is calculated using the sampling frequency and FFT size:
f(k)=k⋅f_s/N_fft 
The higher the spectral centroid, the brighter the sound.

Another related feature is the spectral roll-off, which represents the frequency below which a specific percentage (usually 85%) of the total spectral energy lies (). 
The spectral roll-off frequency for a single audio frame is defined as the frequency bin M which satisfies the inequality:
∑_(k=0)^M▒X (k)≥α⋅∑_(k=0)^(N-1)▒X (k)
Where α = 0.85-0.95
Once M is found, it is converted to Hz:
R=M⋅f_s/N_fft 

The thickness, length and material of strings may also cause observable differences in the sound produced, e.g. for playing E4 in different positions – string 3 will be thicker than string 2 and string 2 will be thicker than string 1. Playing fretted notes will reduce the length of the string. Further variation may be caused for notes such as a G3, which can be played on both nylon (string 3 fret 0) and wound (string 4 fret 5 and string 5 fret 10) strings, since the material is different.
Real strings do not form a perfectly harmonic series because finite string stiffness causes higher vibration modes to deviate from exact integer multiples of the fundamental. The deviation generally increases with mode number, and is influenced by physical parameters including string diameter and vibrating length. This suggests that harmonic-frequency deviations may contain information about the physical string and playing position.
The frequency of the nth partial f_n is given by:
f_n=nf_1 √(1+Bn^2 )
Where B is the inharmonicity coefficient of the string:
B=(π^3 Qd^4)/(64l^2 T)
Where Q is Young’s modulus, d is diameter, l is length, T is tension applied to string.
Measuring the deviation of higher partials from their ideal value could help reveal the string – higher deviation would suggest a higher inharmonicity coefficient, suggesting that the string is thicker (higher string number) or shorter (higher fret number).
Estimating B would require parameters such as Young’s modulus, string diameter, vibrating length and tension. Since the aim of the project is to determine string/fret position from the audio recording alone, these parameters are deliberately not supplied to the classifier. Instead it would be interesting to explore the mean relative harmonic deviation of partial frequencies from the ideal harmonic series as a proxy for inharmonic behaviour.

The temporal envelope (which contains features such as attack, sustain, decay) may also vary based on string shape and material. 
Attack phase:
Shorter strings may produce sharper and faster initial transients because mass displacement is low. Longer strings could have a slightly slower and smoother attack due to more distributed mass.
Thicker strings generally produce a rounder and slower attack. Because it is thicker and under higher tension, there is greater inertia and bending stiffness, so more time is taken for the full displacement to form and for high-frequency transients to peak.
Decay and sustain phase:
Longer strings have longer decay times because the damping rates are lower per cycle (smaller percentage of energy lost during each oscillation compared to shorter string), so sound can last for longer.
Thicker strings usually provide a longer sustain. Because they have more mass, they store much more kinetic energy. The air resistance (viscous damping) that acts on the string is not large enough to quickly deplete this energy reservoir.
Analysing the temporal envelope for attack time, decay time, and the general RMS envelope “shape” for recordings of the same note played in different positions could reveal differences in behaviour. However, this feature is also the most susceptible to plucking variation.

Here is a list of the key factors discussed:
	Harmonics and their amplitudes, amplitude ratios
	Mean relative harmonic deviation from ideal harmonic series
	Spectral centroid
	Rolloff
	Temporal envelope for attack and decay characteristics



# Recording methodology
## Minimising player/equipment-dependent cross-recording variation
The same microphone will be used for all audio recordings. The position of the microphone will be the same across recordings.
A plectrum of thickness 0.71mm will be used for plucking. This is to eliminate plucking variations caused when using the thumb or fingers (e.g. nail shape and length, plucking angle and strength).
The absolute position where the string will be plucked will be marked on the guitar body and the plectrum will be aligned with it. The position will be aligned with the middle of the sound hole. This is to eliminate timbral variations caused by plucking distance along the string, e.g. sul ponticello (closer to the bridge) will produce a brighter and sharper sound, and sul tasto (closer to/over the fretboard) will produce a softer and warmer sound.
All recordings for a particular note (e.g. same note in three different positions) will be taken in the same session to minimize plucking variation across recordings.

## Note selection for proof-of-concept
I decided to intentionally constrain the recording dataset to five notes and 15 string/fret positions (three per note) to test whether physically motivated timbral features can distinguish alternative playing positions before scaling to a larger search space. This allowed controlled experimentation and detailed failure analysis without spending more time taking recordings than analysing the data.
Note	Positions (string, fret)	Reason
E4	(1, 0)
(2, 5)
(3, 9)	chosen to investigate the behaviour of open vs fretted nylon strings
D3	(4, 0)
(5, 5)
(6, 10)	chosen to investigate the behaviour of open vs fretted wound strings
F4	(1, 1)
(2, 6)
(3, 10)	chosen to investigate the behaviour of fretted nylon strings
E3	(4, 2)
(5, 7)
(6, 12)	chosen to investigate the behaviour of fretted wound strings
G3	(3, 0)
(4, 5)
(5, 10)	chosen to investigate the behaviour of nylon vs wound strings

Initially, 5 recordings will be taken for each position for each note. This should be enough to identify anomalous results without using too much time just for recordings.

# Data preprocessing
## File format
All recordings will be recorded in the M4A format with a sampling rate of 48000Hz. since this is the default audio recording format on my smartphone. These recordings will then be converted into WAV to simplify analysis.
 WAV was used because it provides straightforward access to uncompressed PCM samples, avoiding additional codec-decoding steps and lossy-compression artefacts that could affect spectral and harmonic features.

Mono audio will be used instead of stereo to simplify analysis (only need to analyse one data stream instead of 2)
Recording filenames will follow the format:
"{note name}__S{string number}_F{fret number}_{recording number}"

## Metadata
A program (capturing_start_end_data_for_csv.py) will be used to capture and save metadata in a csv file (metadata.csv) for each recording – the note, string, fret, and take/recording will be extracted from the filename.
The onset time and RMS peak time will be captured. This will be useful because each in recording the pluck will not happen at exactly the same time, so these features can help determine where the useful part of the recording (the note) actually starts.
Due to the presence of transients during the initial attack of the note, it may be better to perform analysis during the sustain of the note, when these transients have faded. As an initial rule, the analysis start time will be 0.25s after the RMS peak time. The analysis window will be 0.5s long.
This will constrain the analysis to the part where high-frequency transients and other artefacts that are produced by the attack and/or pluck variation are likely no longer significant, and where the potentially useful parts of the signal have not decayed to the point where they cannot be detected.


