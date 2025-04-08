import lightning
import torch
import os
import requests
import zipfile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TinyImageNet(datasets.ImageFolder):
    def __init__(self, root: str, train: bool = True:
        dataset_root = os.path.join(root, "tiny-imagenet-200")
        if train:
            data_path = os.path.join(dataset_root, "train")
        else:
            data_path = os.path.join(dataset_root, "val")
        
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(74),  
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        
        super().__init__(data_path, transform=transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, label

def create_dataloader(root: str, 
                     train: bool = True, 
                     batch_size: int = 128,
                     num_workers: int = 4):
    dataset = TinyImageNet(root=root, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )