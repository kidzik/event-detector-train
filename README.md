# event-detector

## Workflow

1. Convert c3d to csv with frames as rows and markers/joint angles as columns
2. Run training with different hyperparameters
3. Compare the best models with existing marker-based models

## Code

1. load.py
2. experiments.py
3. Comparison.ipynb

## Extra

Python notebook for training models interactively (with extended comments):
https://github.com/kidzik/event-detector/blob/master/Training.ipynb

Implementation of heursitic based methods:
https://github.com/kidzik/event-detector/blob/master/heuristics.py
