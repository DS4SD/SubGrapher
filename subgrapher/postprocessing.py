#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import numpy as np

# Torch
import torch

# Modules
from subgrapher.utils import compute_overlaps


def apply_threshold(prediction: dict, threshold: float):
    """
    Delete predictions with a confidence score lower than a given cutoff
    """
    nb_objects = prediction["boxes"].size()[0]
    remove_indexes = []
    for index in range(nb_objects):
        if prediction["scores"][index] < threshold:
            remove_indexes.append(index)
    indexes = [e for e in range(nb_objects) if e not in remove_indexes]
    prediction["boxes"] = prediction["boxes"][indexes]
    prediction["masks"] = prediction["masks"][indexes]
    prediction["labels"] = prediction["labels"][indexes]


def revise_predicted_labels(
    prediction: dict, substructures_labels: dict, image_size: tuple
):
    """
    Modify predictions with the following rule on bounding boxes:
        - For each predicted cyclohexene and cyclohexa-x,x+2-diene:
            If its bounding box intersect with the bounding box of a phenyl, a cyclohexene or a cyclohexadiene :
                Change the predicted label to phenyl.
    """
    nb_objects = prediction["boxes"].size()[0]

    for candidate_index in range(nb_objects):
        candidate_label = prediction["labels"][candidate_index].item()

        if (
            candidate_label == substructures_labels["Cyclohexene"]
            or candidate_label == substructures_labels["Cyclohexa-1,3-diene"]
            or candidate_label == substructures_labels["Cyclohexa-1,4-diene"]
        ):
            # Find all phenyls, cyclohexadene and cyclohexadiene
            phenyls_index = []
            for index in range(nb_objects):
                if index != candidate_index:
                    label = prediction["labels"][index].item()
                    if (
                        label == substructures_labels["Phenyl"]
                        or label == substructures_labels["Cyclohexene"]
                        or candidate_label
                        == substructures_labels["Cyclohexa-1,3-diene"]
                        or candidate_label
                        == substructures_labels["Cyclohexa-1,4-diene"]
                    ):
                        phenyls_index.append(index)

            phenyls_boxes = prediction["boxes"][phenyls_index]
            if phenyls_boxes.numel():
                # Extend phenyls, cyclohexadene and cyclohexadiene boxes
                extended_phenyls_boxes = []
                for box in phenyls_boxes:
                    x_min = box[0].item()
                    y_min = box[1].item()
                    x_max = box[2].item()
                    y_max = box[3].item()
                    extended_x_min = max(0, int(x_min - (x_max - x_min) / 5))
                    extended_y_min = max(0, int(y_min - (y_max - y_min) / 5))
                    extended_x_max = min(
                        image_size[0], int(x_max + (x_max - x_min) / 5)
                    )
                    extended_y_max = min(
                        image_size[1], int(y_max + (y_max - y_min) / 5)
                    )
                    extended_phenyls_boxes.append(
                        torch.tensor(
                            [
                                extended_x_min,
                                extended_y_min,
                                extended_x_max,
                                extended_y_max,
                            ]
                        )
                    )
                extended_phenyls_boxes = torch.stack(extended_phenyls_boxes)

                # Check if the current candidate intersect with a phenyl, a cyclohexadene or a cyclohexadiene
                x_min, y_min, x_max, y_max = prediction["boxes"][candidate_index]
                extended_x_min = max(0, int(x_min - (x_max - x_min) / 5))
                extended_y_min = max(0, int(y_min - (y_max - y_min) / 5))
                extended_x_max = min(image_size[0], int(x_max + (x_max - x_min) / 5))
                extended_y_max = min(image_size[1], int(y_max + (y_max - y_min) / 5))
                extended_candidate_box = torch.tensor(
                    [extended_x_min, extended_y_min, extended_x_max, extended_y_max]
                )

                overlaps = compute_overlaps(
                    extended_phenyls_boxes, extended_candidate_box.unsqueeze(0)
                )
                score = np.average(overlaps.max(1))
                if score > 0:
                    # Revise prediction
                    prediction["labels"][candidate_index] = torch.tensor(
                        substructures_labels["Phenyl"]
                    )


def rescale_boxes(
    prediction: dict, image_size: tuple, initial_image_size: tuple, border_size: int
):
    """
    Before running the model, the image undergone some transformations :
        - A resizing, keeping the image scale ratio
        - A padding to create a square image
        - An additionnal padding on borders
    This function reverse these transformations, only for bounding boxes.
    """
    width = image_size[0] - 2 * border_size
    height = image_size[1] - 2 * border_size

    # Remove borders padding
    boxes_without_borders = []
    for box in prediction["boxes"]:
        x_min = box[0].item() - border_size
        y_min = box[1].item() - border_size
        x_max = box[2].item() - border_size
        y_max = box[3].item() - border_size
        boxes_without_borders.append([x_min, y_min, x_max, y_max])

    width_ratio = width / initial_image_size[0]
    height_ratio = height / initial_image_size[1]

    # Rescale and remove padding
    if width_ratio < height_ratio:
        transformed_width = width
        transformed_height = round(width_ratio * initial_image_size[1])
        shift = (height - transformed_height) / 2
        boxes = []
        for box in boxes_without_borders:
            x_min, y_min, x_max, y_max = box
            x_min = x_min / width_ratio
            y_min = (y_min - shift) / width_ratio
            x_max = x_max / width_ratio
            y_max = (y_max - shift) / width_ratio
            boxes.append([x_min, y_min, x_max, y_max])
        prediction["boxes"] = torch.tensor(boxes)

    else:
        transformed_width = round(height_ratio * initial_image_size[0])
        transformed_height = height
        shift = (width - transformed_width) / 2
        boxes = []
        for box in boxes_without_borders:
            x_min, y_min, x_max, y_max = box
            x_min = (x_min - shift) / height_ratio
            y_min = y_min / height_ratio
            x_max = (x_max - shift) / height_ratio
            y_max = y_max / height_ratio
            boxes.append([x_min, y_min, x_max, y_max])
        prediction["boxes"] = torch.tensor(boxes)
