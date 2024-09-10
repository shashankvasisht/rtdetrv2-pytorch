from sympy import im
import torch
import torchvision
import torchvision.transforms.functional as TVF
import os
import numpy as np
from PIL import Image
from typing import Optional, Callable
from osgeo import gdal
import xml.etree.ElementTree as ET

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

import warnings

warnings.filterwarnings("ignore")


import torch
import torch.utils.data as data

from .transforms import (
    convert_to_tv_tensor,
    Compose,
    RandomPhotometricDistort,
    RandomZoomOut,
    RandomIoUCrop,
    SanitizeBoundingBoxes,
    RandomHorizontalFlip,
    Resize,
    ConvertPILImage,
    ConvertBoxes,
)
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
import random
import math
import yaml

################## Example of Albumentations Pipeline ###################
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # Define the augmentation pipeline
# augmentation_pipeline = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
#     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
#     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
#     A.MotionBlur(blur_limit=5, p=0.5),
#     A.MedianBlur(blur_limit=5, p=0.5),
#     A.Blur(blur_limit=5, p=0.5),
#     A.CLAHE(clip_limit=2, p=0.5),
#     A.RandomSizedBBoxSafeCrop(height=512, width=512, p=0.5),
#     A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# # Example usage
# def augment_image(image, bboxes, labels):
#     augmented = augmentation_pipeline(image=image, bboxes=bboxes, labels=labels)
#     return augmented['image'], augmented['bboxes'], augmented['labels']

# # Example data
# image = ...  # Your image here
# bboxes = ...  # Your bounding boxes here in VOC format
# labels = ...  # Your labels here

# # Apply augmentations
# augmented_image, augmented_bboxes, augmented_labels = augment_image(image, bboxes, labels)


#############################################################################
# albumentations equivalent to torchvision transforms

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# tile_size = 256  # Example tile size, replace with your actual value

# train_transforms = A.Compose([
#     A.RandomBrightnessContrast(p=0.5),  # Equivalent to RandomPhotometricDistort
#     A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, 0.5), rotate_limit=0, p=1),  # Equivalent to RandomZoomOut
#     A.RandomSizedBBoxSafeCrop(height=tile_size, width=tile_size, p=0.8),  # Equivalent to RandomIoUCrop
#     A.BboxSafeRandomCrop(erosion_rate=0.0, p=1.0),  # Equivalent to SanitizeBoundingBoxes
#     A.HorizontalFlip(p=0.5),  # Equivalent to RandomHorizontalFlip
#     A.Resize(height=tile_size, width=tile_size),  # Equivalent to Resize
#     A.BboxSafeRandomCrop(erosion_rate=0.0, p=1.0),  # Equivalent to SanitizeBoundingBoxes
#     A.Lambda(image=lambda x, **kwargs: x.astype('float32') / 255.0),  # Equivalent to ConvertPILImage
#     A.Lambda(bboxes=lambda bboxes, **kwargs: [(x, y, w, h) for x, y, w, h in bboxes]),  # Equivalent to ConvertBoxes
#     ToTensorV2()
# ], bbox_params=A.BboxParams(format='coco', label_fields=[]))


class GeoImageryODdata:

    tile_size = 512
    rgb_indices = [0, 1, 2]
    _train_transforms = [
        RandomPhotometricDistort(p=0.5),
        RandomZoomOut(fill=0),
        RandomIoUCrop(p=0.8),
        SanitizeBoundingBoxes(min_size=1),
        RandomHorizontalFlip(),
        Resize(size=[tile_size, tile_size]),
        SanitizeBoundingBoxes(min_size=1),
        ConvertPILImage(dtype="float32", scale=True),
        ConvertBoxes(fmt="cxcywh", normalize=True),
    ]
    _train_policy = {
        "name": "default",
        "epoch": 71,
        "ops": [RandomPhotometricDistort(), RandomZoomOut(), RandomIoUCrop()],
    }
    _val_transforms = [
        Resize(size=[tile_size, tile_size]),
        ConvertPILImage(dtype="float32", scale=True),
    ]
    _val_policy = None

    def __init__(
        self,
        root: str | Path = "data",
        mode: str = "train",  # select between 'train' and 'val'
        num_imgs_per_folder: int = 8000,
        class_mapping_path: str = "class_mapping.yaml",
        # transforms: Optional[Callable] = None,
    ) -> None:

        self.images = []
        self.num_imgs_per_folder = num_imgs_per_folder

        if mode == "train":
            self.transforms = Compose(
                transforms=self._train_transforms, policy=self._train_policy
            )

        else:
            self.transforms = Compose(
                transforms=self._val_transforms, policy=self._val_policy
            )
        if isinstance(root, str):
            root = Path(root)

        folders = [
            root / ix.name
            for ix in root.iterdir()
            if ix.name not in ["cars_pascal_voc", "combined_class_mapping.yaml"]
        ]

        for fx in folders:
            image_fol = fx / "images"
            images = list(image_fol.glob("*.tif"))
            len_images = len(images)
            random.shuffle(images)
            if len_images >= self.num_imgs_per_folder:
                images2append = images[: self.num_imgs_per_folder]
            else:
                multiplier = math.ceil(self.num_imgs_per_folder / len_images)
                images2append = (images * multiplier)[: self.num_imgs_per_folder]
            self.images.extend(images2append)

        # Retrieving class mapping
        # class_mapping_path = root / "combined_class_mapping.yaml"
        with open(class_mapping_path, "r") as file:
            class_mapping = yaml.safe_load(file)
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.images)

    def get_labelpth(self, path):
        if isinstance(path, str):
            path = Path(path)
        label_fol = path.parents[1] / "labels"
        label_suffix = ".xml"
        label_name = str(path.stem) + label_suffix
        label_path = label_fol / label_name
        assert label_path.is_file(), f"{label_path} does not exist."
        return label_path

    def read_image(self, img_path):

        img_file = gdal.Open(img_path)
        rgb_indices = self.rgb_indices

        ## Read image file
        rgbimg = []
        img = []
        for ix in range(img_file.RasterCount):
            b = img_file.GetRasterBand(ix + 1).ReadAsArray()
            if ix in rgb_indices:
                rgbimg.append(b)
            img.append(b)
        # rgbimg = rgbimg[::-1]
        rgbimg = np.dstack(rgbimg)
        img = np.dstack(img)

        rgbimg = Image.fromarray(rgbimg).convert("RGB")

        # return rgbimg, img
        return rgbimg

    def parse_label(self, idx, lbl_path, imgh, imgw):

        tree = ET.parse(lbl_path)
        xmlroot = tree.getroot()
        output = {}
        output["image_id"] = torch.tensor([idx])
        for k in ["area", "boxes", "labels", "iscrowd"]:
            output[k] = []

        for tag_obj in xmlroot.findall("object"):
            bnd_box = tag_obj.find("bndbox")
            xmin, ymin, xmax, ymax = (
                float(bnd_box.find("xmin").text),
                float(bnd_box.find("ymin").text),
                float(bnd_box.find("xmax").text),
                float(bnd_box.find("ymax").text),
            )
            data_class_text = tag_obj.find("name").text

            output["boxes"].append([xmin, ymin, xmax, ymax])
            output["labels"].append(data_class_text)
            output["area"].append((ymax - ymin) * (xmax - xmin))
            output["iscrowd"].append(0)

        w, h = imgw, imgh
        boxes = (
            torch.tensor(output["boxes"])
            if len(output["boxes"]) > 0
            else torch.zeros(0, 4)
        )

        output["boxes"] = convert_to_tv_tensor(
            boxes, "boxes", box_format="xyxy", spatial_size=[h, w]
        )

        output["labels"] = torch.tensor(
            [self.class_mapping[lab] for lab in output["labels"]]
        )

        output["area"] = torch.tensor(output["area"])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])

        return output

    # def resize_img_boxes(image, target, tile_size):
    #     pass

    def __getitem__(self, idx):

        # get image path
        image_pth = self.images[idx]

        # get label path
        label_pth = self.get_labelpth(image_pth)

        # read RGB image
        image = self.read_image(image_pth)

        w, h = image.size

        # parse labels into a dictionary 'target'
        target = self.parse_label(idx, label_pth, h, w)

        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)

        return image, target


# data = GeoImageryODdata(root=r"D:\data\__OTHERDATA__\OD_Foundation_data")


def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
