# LGO-BatchBALD

This repository contains code to reproduce the experiments in the paper under review introducing LGO-BatchBALD. The code was recently refactored and undergoing testing (not core functionality of code but saving of results). Minor (as described before) bug fixes and cleaning of code is an ongoing process and will be complete by the time of the camera-ready submission.

## Requirements

* pytorch
* torchvision
* tqdm
* pylab
* numpy
* pandas
* matplotlib

## Installation

After installing the requirements, clone the repository.

## Usage to reproduce the tables and figures from the submission

All scripts should be run from within the main LGO-BatchBALD directory. Results will be saved in the results subdirectory.

* Tables 2 and 3: run estimators.py
* Tables 4 and 5: run qs_vs_pool.py
* Tables 6 and 7: run timing.py
* Figures 3-6: run visualizations.py
