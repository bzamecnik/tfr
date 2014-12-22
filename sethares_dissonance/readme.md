[GIST](https://gist.github.com/endolith/3066664)

Adaptation of [Sethares' dissonance measurement function](http://sethares.engr.wisc.edu/comprog.html) to Python

Example is meant to match the curve in [Figure 3](http://sethares.engr.wisc.edu/consemi.html#anchor15619672): 

![Figure 3](http://sethares.engr.wisc.edu/images/image1.gif)

Original model used products of the two amplitudes *a1⋅a2*, but this was changed to minimum of the two amplitudes *min(a1, a2)*, as explained in *G: Analysis of the Time Domain Model* appendix of *Tuning, Timbre, Spectrum, Scale*.

> This weighting is incorporated into the dissonance model (E.2) by assuming that the roughness is proportional to the loudness of the beating. ... Thus, the amplitude of the beating is given by the minimum of the two amplitudes.

With the first 6 harmonics at amplitudes 1/n starting at 261.63 Hz, using the product model, it also perfectly matches Figure 4 of [Davide Verotta - Dissonance & Composition](http://davide.gipibird.net/A_folders/Theory/t_dissonance.html), so it should be trustworthy.
