import os
import re
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.search(r'@([^@]+)@', fname)
                if match:
                    label = match.group(1)
                    if label not in self.class_to_idx:
                        idx = len(self.class_to_idx)
                        self.class_to_idx[label] = idx
                        self.idx_to_class[idx] = label
                    self.samples.append((os.path.join(root_dir, fname), self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

dataset = FaceDataset(root_dir='data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(dataset.class_to_idx))
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    correct = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'facenet_classifier.pth')

model.load_state_dict(torch.load('facenet_classifier.pth'))
model.eval()

from PIL import Image

def predict(image_path, model, transform, class_to_idx):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class[pred]

predicted_label = predict('data/7134850@N05_identity_2@7720900376_0.jpg', model, transform, dataset.class_to_idx)
print(f'Predicted label: {predicted_label}')

