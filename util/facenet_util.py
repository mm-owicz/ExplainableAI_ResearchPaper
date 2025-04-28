from torchvision import transforms
from PIL import Image
import os
import re

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

def prepare_image(device, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image


def get_image_for_classes(dir, classes):
    class_files = {}
    for c in classes:
        class_files[c] = []

    for fname in sorted(os.listdir(dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_without_ext = fname.rsplit('.', 1)[0]
            parts = name_without_ext.split('@')
            label = '@'.join(parts[:2])
            if label in classes:
                class_files[label].append(os.path.join(dir, fname))
    return class_files

# classes we trained the model for:
face_classes = ['101506130@N03_identity_0',
 '102962858@N03_identity_2',
 '105391338@N08_identity_0',
 '105546346@N08_identity_2',
 '105700383@N05_identity_7']

CLASSES_FILES = get_image_for_classes('data/', face_classes)

