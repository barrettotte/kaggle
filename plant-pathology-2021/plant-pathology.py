import logging
import random
import sys
import time
from pathlib import Path
from typing import Tuple, List
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

class CONFIG:
    log_level = logging.DEBUG
    data_path = Path('./data')
    model_path = Path('./plant-pathology.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = 42
    train_split = 0.8
    num_workers = 8
    batch_size = 64
    img_size = 256
    epochs = 3

# dataset to load plant images and one hot encode labels
class PlantDataset(Dataset):
    def __init__(self, data: pd.DataFrame, root_dir: Path, transform: v2.Transform = None):
        self.image_ids = data['image'].values
        self.root_dir = root_dir
        self.transform = transform
        self.encoded_labels = self.encode_labels(data['labels'])

    def encode_labels(self, labels: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(MultiLabelBinarizer().fit_transform(labels)).type(torch.LongTensor)

    def load_img(self, idx: int) -> Image.Image:
        return Image.open(self.root_dir / self.image_ids[idx])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.load_img(idx)
        y = self.encoded_labels[idx]

        if self.transform:
            X = self.transform(X)

        return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# init environment
def setup():
    # seed everything for reproducibility
    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    torch.cuda.manual_seed_all(CONFIG.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup logging
    logging.basicConfig(
        level=CONFIG.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('model.log', mode='w'),
        ],
    )

    # log basic env information
    logging.debug(f'torch version: {torch.__version__}')
    logging.debug(f'device: {CONFIG.device}')
    logging.debug(f'{torch.cuda.device_count()} GPU(s) available')
    logging.debug(f'num_workers: {CONFIG.num_workers}')
    logging.debug(f'batch_size: {CONFIG.batch_size}')
    logging.debug(f'img_size: {CONFIG.img_size}')
    logging.debug(f'epochs: {CONFIG.epochs}')

# load training data from csv
def load_data(csv_path: str) -> pd.DataFrame:
    logging.debug(f'Reading data from {csv_path}')
    df = pd.read_csv(csv_path)
    df['labels'] = df['labels'].apply(lambda x: x.split(' '))

    logging.debug(f'Train shape: {df.shape}')
    logging.debug(f'Train columns: {df.columns}')    
    return df

# get distinct classes using one-hot encoding
def get_classes(df: pd.DataFrame) -> List[str]:
    m = MultiLabelBinarizer()
    encoded = pd.DataFrame(m.fit_transform(df['labels']), columns=m.classes_, index=df.index)
    classes = list(encoded.columns)
    logging.debug(f'{len(classes)} classes -> {classes}')
    return classes

# setup training and validation dataloaders
def get_dataloaders(df: pd.DataFrame, root_dir: Path) -> Tuple[DataLoader, DataLoader]:
    transform_mean = (0.485, 0.456, 0.406)
    transform_std = (0.229, 0.224, 0.225)

    transform = v2.Compose([
        v2.Resize(CONFIG.img_size),
        v2.RandomCrop(CONFIG.img_size),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(2),
        # v2.ColorJitter(saturation=(1.0,1.5), contrast=(0.6, 0.8), brightness=(0.85, 0.95)),
        v2.ToTensor(),
        v2.Normalize(mean=transform_mean, std=transform_std),
    ])
    all_ds = PlantDataset(df, root_dir, transform)

    total_samples = len(all_ds)
    train_size = int(total_samples * CONFIG.train_split)
    valid_size = total_samples - train_size
    train_ds, valid_ds = torch.utils.data.random_split(all_ds, [train_size, valid_size])

    logging.info(f'Train dataset: {len(train_ds)} samples')
    logging.info(f'Valid dataset: {len(valid_ds)} samples')

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)
    return (train_loader, valid_loader)

# build model using transfer learning
def build_model(classes: List[str]) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=len(classes)),
        nn.Sigmoid(), # clamp 0-1
    )
    return model.to(CONFIG.device)

# model training step
def train_step(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, log_every: int = 100) -> List[float]:
    losses = []
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(CONFIG.device), targets.to(CONFIG.device)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch % log_every == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            logging.info(f'Training: Loss = {loss:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]')

    return losses

# calculate model accuracy
def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (outputs > threshold).float()
    correct = (preds == targets).sum().item()
    return correct / (targets.size(0) * targets.size(1))

# model validation step
def valid_step(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    total_loss, total_accuracy = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(CONFIG.device), targets.to(CONFIG.device)
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item()
            total_accuracy += calculate_accuracy(outputs, targets)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    logging.info(f'Validation: Accuracy={(100 * avg_accuracy):>0.1f}%, Average_Loss={avg_loss:>8f}')

def main():
    setup()

    # load data
    all_data = load_data(CONFIG.data_path / 'train.csv')
    train_loader, valid_loader = get_dataloaders(all_data, CONFIG.data_path / 'train_images')

    # build model
    classes = get_classes(all_data)
    model = build_model(classes)
    # loss_fn = nn.MultiLabelMarginLoss()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train model
    start_time = time.time()
    logging.debug("Started training model")
    losses = []
    for epoch in range(CONFIG.epochs):
        logging.info(f'Started epoch [{epoch+1}/{CONFIG.epochs}]')
        epoch_losses = train_step(train_loader, model, loss_fn, optimizer)
        valid_step(valid_loader, model, loss_fn)
        losses.extend(epoch_losses)

    logging.debug(f"Model trained after {((time.time() - start_time) / 60):.2f} minute(s)")

    # export model
    logging.info(f"Saved model state dict to {CONFIG.model_path}")
    torch.save(model.state_dict(), CONFIG.model_path)

if __name__ == '__main__':
    main()
