# Quantum-sensing-of-displacements-with-stabilized-GKP-states
Repository with the codes necessary to reproduce the results of the article "Quantum sensing of displacements with stabilized GKP states" by authors L. Labarca, S. Turcotte, A. Blais, B. Royer.

The environement requirements are in "used_packages.txt". 

## The folder figures contains all the data and code to reproduce the figures of the paper. 

For the plots, the basic function calls are in "plotsmodule.py".

## The folder noiseless contains the codes for all results in the absence of noise.

The data is obtained and managed from the jupyter notebooks: "sBs-noiseless", "metrological-potential-noiseless"

All necessary functions are written in the python files "sBs.py", "bits.py", "metrology.py", "backaction.py". 

**sBs-noiseless.ipynb**: used to get the finite-energy sensor states, and probabilities of bitstring measurements. 

**metrological-potential-noiseless**: All metrological analysis are done here.

**sBs.py**: Functions necessary to run sBs

**bits.py**: Functions to manipulate bitstrings. 

**metrology.py**: Functions used in metrological analysis. 

**backaction.py**: Functions used for backaction evading performance analysis.

## The folder Noise contains the codes for all results with noise.

The data is obtained and managed from the jupyter notebooks: "running_noise", "running_backaction_2", "simple-sBs-bitflip-analysis".

All functions are defined in the python files: "sBs", "sBs_noise_analysis", "metrology_noise", "bits", "backaction_noise", "general_functions"

**running_noise.ipynb**: used to create all parameter files, and to run sBs in the presence of noise. Structure is clear within. 

**running_backaction_2**: Idem as above but for backaction evading performance. 

**simple-sBs-bitflip-analysis**: Used to obtain Fig. 16. 

**general_functions.py**: This one contains the functions used to run sBs solving the Lindblad ME. 

**sBs_noise_analysis.py**: Contains all functions that set the code to run, and are arranged here to keep the jupyter notebook running_noise as simple as possible.

The rest contain the functions expected from their title.

## Contact

This github is not the original we used, which became too involved to share. There might be some bugs when attempting to run, if you find this is the case please contact Lautaro at labl2714@usherbrooke.ca to fix the issue.
