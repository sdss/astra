# Payne-Che
The [Payne](https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin) code with [Chebyshev](https://en.wikipedia.org/wiki/Pafnuty_Chebyshev) polynomials

The original Payne code: https://github.com/tingyuansen/The_Payne

The reference paper (Ting et al, 2019): https://doi.org/10.3847/1538-4357/ab2331

This is a modified version of the original Payne code, which estimates stellar parameters based on a normalized spectrum. The modified code 
performs estimation of stellar parameters with a non-normalized spectrum. Instead of normalizing the spectrum, a model spectrum is constructed
that contains the instrumental response function approximated by a Chebyshev polynomial series. The parameters
of the series are searched simultaneously with stellar parameters via optimization routine.

The repository contains modules that perform three different functions:
1. Creating grids of model spectra to be used as training sets;
2. Training neural networks based on grids of model spectra;
3. Fitting models to spectra using trained neural networks.

## Creating grids of model spectra

Run 'quasirandom_grid_single_cell.py' with no arguments. All the parameters are read out from 'random_grid.conf' file.
The parameters define grid extent and step, wavelength grid, the number of models in the grid. The script creates a
quasi-random grid based on a [Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence). For each point in the
quasi-random grid, the script creates a single-cell subgrid that contains this point, by running GSSP code. The model
spectrum for the sampled point is then obtained by linearly interpolating models in the subgrid.

## Training neural networks

Assemble a grid into a single 'npz' file using 'assemble_grid.py' module and run 'train_NN.py' with it. The training algorithm 
uses torch framework and requires a CUDA device. The neural network is saved into 'NN_\*.npz' file.

## Fitting model spectra
Run 'python run_MWMPayne.py <config_file>' to estimate stellar parameters from spectra.

The <config_file> is a text file that contains all the parameters needed to process the data.
Each line has a format "parameter: value, value, ... value". The parameters available are
described below.
```
NN_path: <path to the neural network file>
data_path: <path to the directory with input data>
wave_range: 15000, 17000 # wavelength range in Angstroem
N_chebyshev: 15 # number of Chebyshev polynomials representing instrumental response function
spectral_R: 22500 # resolution of the instrument
N_presearch_iter: 1
N_presearch: 4000
parallel: yes # turns on parallel processing of input spectra
log_dir: <path to the directory where logging data will be saved during processing>
data_format: {APOGEE|HERMES|ASCII} # input data format
```

## Installation

Payne-Che runs on Python 3. Just download or clone the repository.

General package dependencies: **numpy**, **scipy**, **matplotlib**, **astropy**, **multiprocessing**.

Creating grids of models (quasirandom_grid_single_cell.py) requires **shutil**, **subprocess** and [**sobol_seq**](https://github.com/naught101/sobol_seq).

Training neural networks (train_NN.py) requires [**torch**](https://pytorch.org/).




