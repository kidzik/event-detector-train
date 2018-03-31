# event-detector

This software annotates `.c3d` gait trajectories with Heel-Strike and Foot-Off events. It uses neural networks (more precisely Long Short Term Memory networks) through `keras` and `tensorflow` packages.

## Deployed package

This repository contains the training code. If you want to use pretrained version for your data refer to https://github.com/kidzik/event-detector

## Workflow

This repository contains three key procedures:

1. Convert `.c3d` to `.csv` with frames as rows and markers/joint angles as columns
2. Train with different hyperparameters
3. Compare the best models with existing marker-based models

These procedures are implemented in the following files respectively:

1. load.py
2. experiments.py
3. Comparison.ipynb

## Extra

Python notebook for training models interactively (with extended comments):
https://github.com/kidzik/event-detector-train/blob/master/Training.ipynb

Implementation of heursitic based methods:
https://github.com/kidzik/event-detector-train/blob/master/heuristics.py

