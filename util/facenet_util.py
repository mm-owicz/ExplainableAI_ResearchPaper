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
            match = re.search(r'@([^@]+)@', fname)
            if match:
                label = match.group(1)
                if label in classes:
                    class_files[label].append(os.path.join(dir, fname))
    return class_files

# classes we trained the model for:
face_classes = ['N08_identity_4',
 'N00_identity_14',
 'N00_identity_11',
 'N00_identity_0',
 'N04_identity_5']

CLASSES_FILES = get_image_for_classes('data/', face_classes)

