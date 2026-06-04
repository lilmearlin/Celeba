import os
import torch #для нейронки
import numpy as np #для векторов
from PIL import Image #для работы с картинками
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms #для нейронки
from facenet_pytorch import InceptionResnetV1 #модели для обнаружения лиц
# Настройка устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")