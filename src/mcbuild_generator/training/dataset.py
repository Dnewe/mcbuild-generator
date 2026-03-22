from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
from typing import List


class BuildDataset(Dataset):
    def __init__(self, filepaths) -> None:
        super().__init__()
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        return torch.load(self.filepaths[index]).type(torch.long)


def custom_collate_fn(batch):
    """
    Custom collate function to prevent stacking of tensors with different shapes
    """
    return list(batch)


def get_dataset(data_dir: str, train_val_split: List[float] = [0.8, 0.2]):
    filepaths = [
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.split(".")[-1] == "pt"
    ]
    dataset = BuildDataset(filepaths)

    train_dataset, val_dataset = random_split(dataset, lengths=train_val_split)
    return train_dataset, val_dataset


def get_loaders(
    data_dir: str, train_val_split: List[float], batch_size: int, num_workers: int = 0
):
    """Create train and validation dataloaders."""
    train_dataset, val_dataset = get_dataset(data_dir, train_val_split)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader
