import os
import torch #для нейронки
import numpy as np #для векторов
from PIL import Image #для работы с картинками
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms #для нейронки
from facenet_pytorch import InceptionResnetV1 #модели для обнаружения лиц

class CelebaDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted([x for x in os.listdir(image_dir) if x.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)), #приводит картинку к размеру 160 на 160
            transforms.ToTensor(), #из пикселей в тензор PyTorch
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])#центрирование данных
        ])

