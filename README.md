This repository contains the pipeline to analyse GRBs from Fermi/GBM and compute the minimum variability timescale, the evolution
of the four properties listed in Maccary et al. 2026, and the position of a given burst in the E_pi-E_iso plane (Maccary et al. 2026, in preparation). This pipeline can be used to quickly obtain indices of merger origin from the prompt-emission time profile to incite for a multi-wavelength follow-up campaign of these events.

This pipeline is written in Python and is using standard Python libraries. To use this Github repository, you need to install the GBM data tools: https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gdt-docs/ and to install the ultimate version of MEPSA https://www.fe.infn.it/u/guidorzi/new_guidorzi_files/code.html that is described in Maistrello et al. 2026.

The pipeline allows to:

- Download the data from the Fermi/GBM database;
- Perform the subtraction of the background;
- Apply MEPSA to the background subtracted light curves;
- Compute the minimum variability timescale;
- If the redshift of the source is known, to obtain the position in the Epi-Eiso of the burst and compute the likelihood of this position being compatible with the Epi-Eiso relation for type-II GRBs. Equivalently if the redshift is unknown one can obtain the track of the burst in the Amati plane as a function of redshift as done in Maccary et al. 2026.
