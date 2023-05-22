# pyPartAnalysis

Package for processing ImpactT and ImpactZ particle distributions as well as performing analysis on particle distributions in pandas dataframes.

## Installation

The package can be installed from PyPI, using the following command:

```bash
pip install py-particle-analysis
```

## Compatibility

Compatible with Python 3.9.7. Refer to the requirements.txt for the dependencies.

Works with the default NERSC python module as of March 13th, 2023.

# Example Use Cases for ImpactT/Z

Example Jupyter notebooks using ImpactT and ImpactZ output files can by found [here](examples/). This package can be useful for creating interesting visualizations of from the particle distributions, such as the ones below:

<p align='center'><img src=".\examples\1-ImpactT_distribution\NormalizedPhaseSpaceVsSlice.png"></p>
<p align='center'><em>Series of z slices along the normalized x phase space.</em></p>

<p align='center'><img src=".\examples\4-Bunching Factor\bunching_area.gif"></p>
<p align='center'><em>Transversely binned bunching factor vs wavelength.</em></p>

# Using on NERSC

### Editable Installation with Custom Environment

As the performance of computing environements on supercomputers like NERSC can be heavily dependent on the particular library that is installed, it is recommended to install the dependencies with `conda`. Since we will be installing our own packages, a custom environment should be used as is describe [here](https://docs.nersc.gov/development/languages/python/nersc-python/#option-2-module-conda-activate). To build on top of the existing python module:

```bash
module load python
conda create --name myenv
conda activate myenv
```

The user could instead clone the default environment given by `module load python`. To check if this changes in the future, look at the `setenv` path that appears when `module show python` is called. Cloning the environment is not necessary but the commands are given below:

Cloning environment https://docs.nersc.gov/development/languages/python/nersc-python/#using-conda-clone

```bash
module load python
conda create --name myenv --clone /global/common/software/nersc/cori-2022q1/sw/python/3.9-anaconda-2021.11
conda activate myenv
```

The user can then install the pyPartAnalysis package by cloning the repository as is shown below:

```bash
git clone https://github.com/lm314/py-particle-analysis.git
cd py-particle-analysis
conda install --file requirements.txt
conda install ipykernel
conda develop src/python
```

Note that we install `ipykernel` as well so that we can use the environment with JupyterLab. If that is not desired, then that command can be excluded. Any already present packages that already meet the requirements will be skipped over.

To incorporate the custom environment into JupyterLab on NERSC, we follow the directions [here](https://docs.nersc.gov/services/jupyter/#conda-environments-as-kernels). If the commands above have already been executed, then one need only run the following command

```bash
python -m ipykernel install --user --name myenv --display-name MyEnv
```

The `myenv` and `myEnv` environment names can be changed to what the user desires.

### Without Custom Environment

If the user does not wish to use a custom environment, then the Git repository can be cloned 

```bash
git clone https://github.com/lm314/py-particle-analysis.git
```

and the `pyPartAnalysis` package (`./py-particle-analysis/src/pyPartAnalysis`) can be copied into the directory where it is called, whether by a script or a jupyter notebook.

If the user wants to use the package across multiple directories, they can make the `pyPartAnalysis` module available on the Python and Jupyter path, add the following lines to the `.bashrc` file in the home directory:

```bash
export PYTHONPATH='/global/homes/firstletterofusername/username/py-particle-analysis'
export JUPYTER_PATH='/global/homes/firstletterofusername/username/py-particle-analysis'
```

Note that we assume that the directory is in the home directory of 'username', where username is replaced with your name; though the path can be to wherever the module is eventually stored.

# Using on ASU Agave

Refer to the documentation [here](https://asurc.atlassian.net/wiki/spaces/RC/pages/125829137/Using+anaconda+environments) for creating a custom environment.


