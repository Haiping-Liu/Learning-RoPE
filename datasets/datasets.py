import lightning
import torch
import os
import requests
import zipfile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

class TrainDataset(datasets.ImageFolder):
    def __init__(self, data_path: str):

        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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

class ValDataset(Dataset):
    def __init__(self, data_path: str):
        # 1. 路径设置
        annotations_file = os.path.join(data_path, 'val_annotations.txt')
        self.images_dir = os.path.join(data_path, 'images')

        # 2. 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize(74),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 3. 加载标注信息
        self.annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                self.annotations.append((parts[0], parts[1]))  # (filename, label_name)

        # 4. 构造类别名称 → index 的映射
        all_classes = sorted(list(set(label for _, label in self.annotations)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, label_name = self.annotations[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 读取图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 将 label 转为 tensor
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        return image, label

def create_dataloader(data_path: str, 
                     train: bool = True, 
                     batch_size: int = 128,
                     num_workers: int = 4):
    dataset = TrainDataset(os.path.join(data_path, 'train')) if train else ValDataset(os.path.join(data_path, 'val'))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )