# src/un_detector/data/datasets.py
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class HazmatDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # Load annotations
        with open(annotations_file) as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']

        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.ids = list(self.images.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = os.path.join(self.data_dir, 'images', img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Get annotations
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            bbox = ann['bbox']
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]

            # Filter out degenerate boxes
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
