# Setup

Notes on setting up local environment

TODO: could this be automated?

## General

Install conda
- Windows: Just use installer - https://www.anaconda.com/download/

## Jupyter Env

```sh
conda create --name kaggle
conda info --envs
conda activate kaggle

conda update --all -c conda-forge
conda install -c conda-forge jupyterlab jedi ipython ipykernel sqlite

jupyter lab
```

## PyTorch

Base pytorch packages installed

```sh
conda install -c numpy pandas matplotlib scikit-learn -y
conda install -c conda-forge torchinfo torchmetrics -y

# https://pytorch.org/get-started/locally/#anaconda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```
