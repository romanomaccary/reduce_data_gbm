This repository contains the pipeline to analyse GRBs from Fermi/GBM and compute the minimum variability timescale, the evolution
of the four properties listed in Maccary et al. 2026, and the position of a given burst in the E_pi-E_iso plane (Maccary et al. 2026, in preparation). This pipeline can be used to quickly obtain indices of merger origin from the prompt-emission time profile to incite for a multi-wavelength follow-up campaign of these events.

This pipeline is written in Python and uses standard Python libraries. To use this Github repository, you need to install the GBM data tools: https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gdt-docs/ and the ultimate version of MEPSA https://www.fe.infn.it/u/guidorzi/new_guidorzi_files/code.html that is described in Maistrello et al. 2026.

This pipeline allows to:

- Download the data from the Fermi/GBM database indicating the Fermi ID of the burst;
- Choosing the detector units that triggered the event;
- Perform the subtraction of the background;
- Apply MEPSA to the background subtracted light curves;
- Compute the minimum variability timescale;
- Obtain the temporal evolution of the peak rates, waiting times, FWHMs, and peak energy, as done in Maccary et al. 2026;
- If the redshift of the source is known, to obtain the position in the Epi-Eiso of the burst and compute the likelihood of this position being compatible with the Epi-Eiso relation for type-II GRBs. Equivalently if the redshift is unknown one can obtain the track of the burst in the Amati plane as a function of redshift, as done in Maccary et al. 2026.

A module to perform time-resolved and time-integrated spectral analysis is provided.

