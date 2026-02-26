# FER-2013

A deep learning architecture for classifying facial emotions, learning from FER-2013 dataset.

## Project Structure

```bash
fer-2013
.
├── checkpoints
├── configs
│   └── config.yaml
├── data
│   └── raw
│       ├── test
│       └── train
├── environment.yaml
├── LICENSE
├── main.py
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_evaluation.ipynb
├── README.md
├── src
│   ├── config.py
│   ├── data
│   │   ├── dataset.py
│   │   ├── fetch_data.py
│   │   └── __init__.py
│   ├── focal_loss
│   │   ├── focal_loss.py
│   │   └── __init__.py
│   └── model
│       ├── callbacks.py
│       ├── eval.py
│       ├── __init__.py
│       ├── model.py
│       └── train.py
└── tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate fer-2013
```

## Run

```bash
python -m main
```

## License

This project is under MIT license [LICENSE](./LICENSE).
