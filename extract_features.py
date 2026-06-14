import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CelebADataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted([x for x in os.listdir(image_dir) if x.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def extract_and_save_embeddings(image_dir, save_embeddings_path, save_names_path, batch_size=64):
    dataset = CelebADataset(image_dir=image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    all_image_names = []
    
    print(f"Начинаем извлечение цифровых паспортов для {len(dataset)} лиц...")
    
    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(dataloader):
            images = images.to(device)
            embeddings = model(images)
            embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_image_names.extend(names)
            
            if batch_idx % 10 == 0:
                print(f"Обработано батчей: {batch_idx} / {len(dataloader)}")

    all_embeddings = np.vstack(all_embeddings)
    np.save(save_embeddings_path, all_embeddings)
    
    with open(save_names_path, 'w') as f:
        for name in all_image_names:
            f.write(f"{name}\n")
            
    print(f"Успешно! База векторов сохранена в {save_embeddings_path}")
    print(f"Список имён сохранен в {save_names_path}")


model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if __name__ == "__main__":
    IMAGE_DIRECTORY = r"C:\celeba_\img_align_celeba"
    EMBEDDINGS_OUTPUT = "all_embeddings.npy"
    NAMES_OUTPUT = "image_names.txt"
    
    extract_and_save_embeddings(IMAGE_DIRECTORY, EMBEDDINGS_OUTPUT, NAMES_OUTPUT, batch_size=64)