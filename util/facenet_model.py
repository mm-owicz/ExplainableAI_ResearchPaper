import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_classes=10):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        label_names = []

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.search(r'@([^@]+)@', fname)
                if match:
                    label = match.group(1)
                    if label not in label_names:
                        label_names.append(label)

        label_names = label_names[:max_classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(label_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.search(r'@([^@]+)@', fname)
                if match:
                    label = match.group(1)
                    if label in self.class_to_idx:
                        path = os.path.join(root_dir, fname)
                        self.samples.append((path, self.class_to_idx[label]))
        self.labels = label_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# limit do 5 klas - długo zajmuje trening :c
dataset = FaceDataset(root_dir='data/', transform=transform, max_classes=5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(dataset.class_to_idx))
