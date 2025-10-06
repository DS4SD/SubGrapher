#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

# Data transformation
import albumentations as albu

# Images
import cv2

# Mathematics
import numpy as np
from albumentations.augmentations import functional as F

# Standard Downscale and GridDistortion
from albumentations.augmentations.geometric.transforms import GridDistortion
from albumentations.augmentations.transforms import Downscale
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

# Modified GridDistortion, not compatible with newer albumentation versions

# class Downscale(DualTransform):
#     """
#     Reduce resolution of the image and masks without modifying bounding boxes.
#     """
#     def __init__(self, scale_min=0.25, scale_max=0.25, interpolation=cv2.INTER_NEAREST, always_apply=False, p=0.5):
#         super(Downscale, self).__init__(always_apply, p)
#         if scale_min > scale_max:
#             raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
#         if scale_max >= 1:
#             raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
#         self.scale_min = scale_min
#         self.scale_max = scale_max
#         self.interpolation = interpolation

#     def apply(self, image, scale, interpolation, **params):
#         return F.downscale(image, scale=scale, interpolation=interpolation)

#     def get_params(self):
#         return {
#             "scale": np.random.uniform(self.scale_min, self.scale_max),
#             "interpolation": self.interpolation,
#         }

#     def apply_to_mask(self, mask, scale, interpolation, **params):
#         return F.downscale(mask, scale=scale, interpolation=interpolation)

#     def apply_to_bbox(self, bbox, scale, interpolation=cv2.INTER_LINEAR, **params):
#         return bbox


# Modified GridDistortion, not compatible with newer albumentation versions.

# from albumentations.core.transforms_interface import DualTransform, to_tuple, ImageOnlyTransform

# class GridDistortion(DualTransform):
#     """
#     """

#     def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, p=0.5):
#         super(GridDistortion, self).__init__(always_apply, p)
#         self.num_steps = num_steps
#         self.distort_limit = to_tuple(distort_limit)
#         self.interpolation = interpolation
#         self.border_mode = border_mode
#         self.value = value
#         self.mask_value = mask_value

#     def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
#         return F.grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode, self.value)

#     def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
#         return F.grid_distortion(img, self.num_steps,stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

#     def get_params(self):
#         stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
#         stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
#         return {"stepsx": stepsx, "stepsy": stepsy}

#     def get_transform_init_args_names(self):
#         return ("num_steps", "distort_limit", "interpolation", "border_mode", "value", "mask_value")

#     def apply_to_bbox(self, bbox, interpolation=cv2.INTER_LINEAR, **params):
#         """
#         Useful only for visibility
#         """
#         x_min, y_min, x_max, y_max = bbox
#         new_bbox = tuple([(x_min*params["rows"] - (params["rows"]/(2*self.num_steps)))/params["rows"], (y_min*params["cols"] - (params["cols"]/(2*self.num_steps)))/params["cols"], \
#                     (x_max*params["rows"] + (params["rows"]/(2*self.num_steps)))/params["rows"], (y_max*params["cols"] + (params["cols"]/(2*self.num_steps)))/params["cols"]])
#         return new_bbox


class PixelSpotNoise(ImageOnlyTransform):
    """Apply pixel noise to the input image.
    Args:
        value ((float, float, float) or float): color value of the pixel.
        prob (float): probability to add pixels
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_holes=1,
        max_holes=8,
        min_height=0.1,
        max_height=0.8,
        min_width=0.1,
        max_width=0.8,
        prob=0.05,
        value=0,
        always_apply=False,
        p=0.5,
    ):
        super(PixelSpotNoise, self).__init__(always_apply, p)
        self.prob = (prob,) * 2
        self.value = value
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

    def apply(self, img, **params):
        prob = (self.prob[1] - self.prob[0]) * np.random.random_sample() + self.prob[0]
        holes = self._get_hole_list(img.shape[:2])
        for x1, y1, x2, y2 in holes:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if np.random.random_sample() <= prob:
                        img[y, x] = self.value
        return img

    def get_transform_init_args(self):
        return {"value": self.value, "prob": self.prob}

    def _get_hole_list(self, img_shape):
        height, width = img_shape[:2]
        holes = []
        for _n in range(np.random.randint(self.min_holes, self.max_holes)):
            hole_height = int(
                height * np.random.uniform(self.min_height, self.max_height)
            )
            hole_width = int(width * np.random.uniform(self.min_width, self.max_width))
            # Offset to ensure the image borders remain white
            y1 = np.random.randint(10, height - hole_height - 10)
            x1 = np.random.randint(10, width - hole_width - 10)
            holes.append((x1, y1, x1 + hole_width, y1 + hole_height))
        return holes


def get_transforms(phase, image_size):
    """
    Get transforms to apply on fly to images.
    For training, four severities of transforms are proposed, in this way, a suitable transform can be applied regarding the molecule complexity.
    Only spatial transformations are applied on the image and masks. Masks are classification targets and should not have changed values.
    """
    if phase == "test":
        transforms = {}
        transforms_list = []
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["clean"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )
        transforms["soft"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )
        transforms["medium"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )
        transforms["hard"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )
        transforms["extreme"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )
        return transforms

    if phase == "train":
        transforms = {}

        # Clean
        transforms_list = []
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["clean"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Soft
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.7, 0),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["soft"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Medium
        transforms_list = []
        # border_mode=cv2.BORDER_REPLICATE isn't working unfortunately
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.5, 0),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        # cv2.INTER_LANCZOS4 for non binary images
        # Applied to the image and masks
        transforms_list.append(
            Downscale(scale_min=0.4, scale_max=0.99, p=0.7)
        )  # Breaks visibility check-ups
        # Applied to the image, its masks and bounding boxes
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        # min_visibility = 0.99 get rid of bounding boxes which are partially moved out of the observation window with augmentations
        # attached labels and masks that should be removed are not modified
        transforms["medium"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Hard
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.2, 0),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(Downscale(scale_min=0.2, scale_max=0.75, p=0.7))
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["hard"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme (Grid distortions require a zoom out)
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.7, -0.5),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )  # (-0.2, 0)
        transforms_list.append(Downscale(scale_min=0.7, scale_max=0.99, p=0.7))
        transforms_list.append(
            GridDistortion(
                distort_limit=(-0.15, 0.15),
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
                p=0.5,
            )
        )
        transforms_list.append(
            PixelSpotNoise(
                min_holes=1,
                max_holes=5,
                min_height=0.1,
                max_height=0.8,
                min_width=0.1,
                max_width=0.8,
                prob=0.05,
                value=0,
                always_apply=False,
                p=0.5,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["extreme"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme (for fingerprint retrieval evaluation) 2/4
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.7, -0.5),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(Downscale(scale_min=0.7, scale_max=0.99, p=0.7))
        transforms_list.append(
            GridDistortion(
                distort_limit=(-0.15, 0.15),
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
                p=0.5,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["fingerprint-eval"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme variant 2 (for fingerprint retrieval evaluation ablation) 4/5
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=15,
                scale_limit=(-0.8, -0.6),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(Downscale(scale_min=0.8, scale_max=0.99, p=0.7))
        transforms_list.append(
            GridDistortion(
                distort_limit=(-0.2, 0.2),
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
                p=0.5,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["fingerprint-eval-2"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme variant 3 (for fingerprint retrieval evaluation ablation) 5/5
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=20,
                scale_limit=(-0.9, -0.7),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(Downscale(scale_min=0.9, scale_max=0.99, p=0.7))
        transforms_list.append(
            GridDistortion(
                distort_limit=(-0.25, 0.25),
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
                p=0.5,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["fingerprint-eval-3"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme variant 4 (for fingerprint retrieval evaluation ablation) 2/5
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=10,
                scale_limit=(-0.4, -0.3),
                shift_limit=0.01,
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(Downscale(scale_min=0.5, scale_max=0.8, p=0.7))
        transforms_list.append(
            GridDistortion(
                distort_limit=(-0.10, 0.10),
                border_mode=cv2.BORDER_CONSTANT,
                value=(1, 1, 1),
                mask_value=(0, 0, 0),
                interpolation=cv2.INTER_NEAREST,
                p=0.5,
            )
        )
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["fingerprint-eval-4"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        # Extreme variant 5 (for fingerprint retrieval evaluation ablation) 1/5
        transforms_list = []
        # transforms_list.append(albu.ShiftScaleRotate(rotate_limit=10, scale_limit=(-0.4, -0.3), shift_limit=0.01, p=0.9, \
        #                                              border_mode=cv2.BORDER_CONSTANT, value=(1,1,1), mask_value=(0,0,0), interpolation=cv2.INTER_NEAREST))
        # transforms_list.append(Downscale(scale_min=0.5, scale_max=0.8, p=0.7))
        # transforms_list.append(GridDistortion(distort_limit=(-0.10, 0.10), border_mode=cv2.BORDER_CONSTANT, value=(1, 1, 1), mask_value=(0,0,0), interpolation=cv2.INTER_NEAREST, p=0.5))
        transforms_list.append(albu.Resize(image_size[0], image_size[1]))
        transforms["fingerprint-eval-5"] = albu.Compose(
            transforms_list,
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.999999,
            ),
        )

        return transforms
