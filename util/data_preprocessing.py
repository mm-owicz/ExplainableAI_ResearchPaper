import os
import pandas as pd
from sklearn.model_selection import train_test_split

from cnn_parameters import TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, IMAGE_LABEL, CLASS_LABEL, DOG_LABEL, CAT_LABEL, VALID_SIZE


train_images = os.listdir(TRAIN_IMAGES_PATH)
test_images = os.listdir(TEST_IMAGES_PATH)

classes = [DOG_LABEL if DOG_LABEL in filename else CAT_LABEL for filename in train_images]
train_set_df = pd.DataFrame({IMAGE_LABEL: train_images, CLASS_LABEL: classes})

train_df, val_df = train_test_split(train_set_df, test_size=VALID_SIZE, stratify=train_set_df[CLASS_LABEL], random_state=42)
