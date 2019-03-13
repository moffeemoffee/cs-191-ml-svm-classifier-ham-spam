# Setup

## Folder structure

```
├── dataset
|  └── trec07p
|     ├── data
|     ├── delay
|     ├── full
|     ├── partial
|     └── README.txt
├── gridsearch-log.txt
├── gridsearch.py
├── helper.py
├── preprocess.py
├── processed.csv
├── README.md
├── requirements.txt
├── scores.csv
└── train.py
```

Get the data set from https://plg.uwaterloo.ca/~gvcormac/treccorpus07/ and extract it into the dataset folder, following the provided structure.

## Pre-requisites

Python 3.7.2 (un-tested on other versions)

## Installation

```
pip install -r requirements.txt
```

## Usage

Run `train.py`, it uses `processed.csv`:
```
python train.py
```

You can also run `gridsearch.py` to check the results of the grid search, or attempt different configurations.

Also, you can generate a new `processed.csv` if you want with `preprocess.py`:
```
python preprocess.py
```