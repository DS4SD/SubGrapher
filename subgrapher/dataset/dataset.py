#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Data Science
import torch
from PIL import Image
from torch.utils.data import Dataset

# Modules
from subgrapher.utils import crop_tight, resize_image


class ImagesDatasetInference(Dataset):
    def __init__(
        self, images_paths, image_size, border_size=50, binarization_threshold=0.6
    ):
        self.images_paths = images_paths
        self.image_size = image_size
        self.binarization_threshold = binarization_threshold
        self.border_size = border_size

    def __getitem__(self, index):
        # Get the image
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert("RGB")

        # Remove borders
        pil_image = crop_tight(image)

        # Resize, add small borders
        pil_image = resize_image(pil_image, self.image_size, self.border_size)

        # Threshold and convert to float
        image = np.array(pil_image, dtype=np.float32) / 255
        image[image > self.binarization_threshold] = 1.0
        image[image != 1.0] = 0.0
        image = np.stack((image,) * 3, axis=-1)

        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, image_path

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.images_paths)
