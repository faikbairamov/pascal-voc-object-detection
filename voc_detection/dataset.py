from torchvision.transforms.functional import to_tensor
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(VOC_CLASSES, start=1)
}

class VOCDetectionDataset(Dataset):
    def __init__(self, dataset_dir, split_file, horizontal_flip_prob=0.0):
        self.dataset_dir = Path(dataset_dir)
        self.image_ids = Path(split_file).read_text().splitlines()
        if not 0 <= horizontal_flip_prob <= 1:
            raise ValueError("Flip probability must be between 0 and 1.")
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = self.dataset_dir / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = self.dataset_dir / "Annotations" / f"{image_id}.xml"

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        annotation = ET.parse(annotation_path).getroot()

        boxes = []
        labels = []
        difficult = []

        for obj in annotation.findall("object"):
            box = obj.find("bndbox")

            boxes.append([
                float(box.findtext("xmin")) - 1,
                float(box.findtext("ymin")) - 1,
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            ])

            labels.append(CLASS_TO_ID[obj.findtext("name")])
            difficult.append(int(obj.findtext("difficult", "0")))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "difficult": torch.tensor(difficult, dtype=torch.bool),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }

        tensor = to_tensor(image)
        if self.horizontal_flip_prob and torch.rand(()) < self.horizontal_flip_prob:
            tensor = tensor.flip(-1)
            boxes = target["boxes"]
            boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
        return tensor, target

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
