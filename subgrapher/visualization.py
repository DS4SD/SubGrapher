#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Python standard library
from collections import defaultdict

import matplotlib.patches as patches

# Plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Maths
import torch
from matplotlib import colors
from matplotlib.colors import LogNorm

# Modules
from subgrapher.utils import compute_overlaps


def plot_boxes_and_scores(
    prediction,
    plot,
    colors,
    labels_substructures,
    caption=False,
    caption_file=None,
    prefix="",
    display_large_molecule=False,
    overwrite_caption_file=False,
):
    if caption:
        if caption_file:
            if not (os.path.exists(caption_file)) or overwrite_caption_file:
                file = open(caption_file, "w")
            else:
                file = open(caption_file, "a")
        else:
            print("Caption")

    display_index = 0
    for index, box in enumerate(prediction["boxes"]):
        x_min, y_min, x_max, y_max = box
        height = x_max - x_min
        width = y_max - y_min
        score = prediction["scores"][index].item()
        color = colors[prediction["labels"][index] - 1]
        position = (x_max, y_max)
        if display_large_molecule:
            rectangle = patches.Rectangle(
                (x_min, y_min),
                height,
                width,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
        else:
            rectangle = patches.Rectangle(
                (x_min, y_min),
                height,
                width,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        plot.add_patch(rectangle)
        if display_large_molecule:
            plt.text(
                x=position[0],
                y=position[1],
                s=prefix + f"{display_index} ({round(score, 4)})",
                color="black",
                bbox=dict(facecolor="white", edgecolor=color, pad=2),
            )
        else:
            plt.text(
                fontsize=20,
                x=position[0],
                y=position[1],
                s=prefix + f"{display_index} ({round(score, 4)})",
                color="black",
                bbox=dict(facecolor="white", edgecolor=color, pad=3),
            )
        if caption:
            if caption_file:
                file.write(
                    prefix
                    + f"{display_index} : {labels_substructures[prediction['labels'][index].item()]} \n"
                )
            else:
                print(
                    prefix
                    + f"{display_index} : {labels_substructures[prediction['labels'][index].item()]}"
                )

        display_index += 1

    if caption_file:
        file.close()


def plot_boxes(
    data,
    plot,
    colors,
    labels_substructures,
    caption=True,
    style="-",
    display_indices=True,
    alpha=1,
):
    """
    inputs - data is target or prediction
    """
    if caption:
        print("Caption")
    for index, box in enumerate(data["boxes"]):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        color = colors[data["labels"][index] - 1]
        if index % 4 == 0:
            position = (x_min, y_min)
        elif index % 3 == 0:
            position = (x_min, y_max)
        elif index % 2 == 0:
            position = (x_max, y_min)
        else:
            position = (x_max, y_max)
        rectangle = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=1,
            linestyle=style,
            edgecolor=color,
            facecolor="none",
        )
        plot.add_patch(rectangle)
        if display_indices:
            plt.text(
                x=position[0],
                y=position[1],
                s=f"{index}",
                color="black",
                bbox=dict(facecolor="white", edgecolor=color, pad=3),
            )
        if caption:
            print(f"{index} : {labels_substructures[data['labels'][index].item()]}")


def plot_masks(data):
    """
    inputs - data is target or prediction
                - target["masks"] is a UInt8Tensor[N, H, W]
                - prediction["masks"] is a UInt8Tensor[N, 1, H, W]
    """
    plot_positions = [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ]
    for index, mask in enumerate(data["masks"]):
        if index < 12:
            plot = plt.subplot2grid((4, 4), plot_positions[index], rowspan=1, colspan=1)
            if len(mask.size()) == 3:
                plot.imshow(mask[0], cmap="Greys_r")
            else:
                plot.imshow(mask, cmap="Greys_r")


def display_fingerprint_matrix(matrix, visualize_visual_fingerprint_path):
    valid_rows = ~np.all(matrix == 0, axis=1)
    filtered_matrix = matrix[valid_rows][:, valid_rows]

    original_row_indices = np.arange(matrix.shape[0])[valid_rows]
    original_col_indices = np.arange(matrix.shape[1])[valid_rows]

    annot_matrix = np.empty_like(filtered_matrix, dtype=object)
    annot_matrix[filtered_matrix != 0] = np.round(
        filtered_matrix[filtered_matrix != 0], 1
    )
    annot_matrix[~(filtered_matrix != 0)] = ""

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        filtered_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="viridis",
        norm=LogNorm(
            vmin=np.min(filtered_matrix[filtered_matrix > 0]),
            vmax=np.max(filtered_matrix),
        ),
        xticklabels=original_col_indices,
        yticklabels=original_row_indices,
        annot_kws={"size": 16},  # 10 For large display: annot_kws = {"size": 16}
    )
    # For large display
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(visualize_visual_fingerprint_path, dpi=300)


def display_predictions(
    image, predictions, subgrapher_config, colors, path, caption_path
):
    # Adjust mappings
    labels_substructures = {}
    substructures_labels = {}
    for substructure, index in subgrapher_config["fg_substructures_labels"].items():
        labels_substructures[index] = substructure
        substructures_labels[substructure] = index
    for substructure, index in subgrapher_config["cb_substructures_labels"].items():
        labels_substructures[
            index + len(subgrapher_config["fg_substructures_labels"])
        ] = substructure
        substructures_labels[substructure] = index + len(
            subgrapher_config["fg_substructures_labels"]
        )

    # Convert prediction format
    predictions = {
        "boxes": torch.tensor([a["bbox"] for a in predictions]),
        "labels": torch.tensor(
            [substructures_labels[a["substructure"]] for a in predictions]
        ),
        "scores": torch.tensor([a["confidence"] for a in predictions]),
    }

    # Display
    plt.figure(figsize=(28, 28))
    plot = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    plot.imshow(image.permute(1, 2, 0), cmap="Greys_r")
    plot_boxes_and_scores(
        predictions,
        plot,
        colors,
        labels_substructures,
        caption=True,
        caption_file=caption_path,
        overwrite_caption_file=True,
    )
    plt.savefig(path)
    plt.close()
