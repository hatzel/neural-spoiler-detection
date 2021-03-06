# Neural Spoiler Detection

This repository contains the code used for spoiler detection in the thesis "Detecting Spoilers using Neural Networks".
It is largely based on [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/).

To run training set up a virtual environment using `python -m virtualenv venv`, activate it using `source venv/bin/activate` and install dependencies using `pip install -r requirements.txt`

## Implementation of Segmentation Metrics
There exist NLTK based implementations of WindowDiff and WinPR.
For this thesis however a custom implementation was created and tested (see `src/metrics.py` and `tests/test_{windowdiff,winpr}.py`).
