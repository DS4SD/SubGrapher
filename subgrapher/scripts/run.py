#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from collections import defaultdict

# Library
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from subgrapher.dataset.dataset import ImagesDatasetInference

# Modules
from subgrapher.model import SubGrapher
from subgrapher.utils import get_mappings
from subgrapher.visualization import display_fingerprint_matrix, display_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subgrapher-config-path",
        type=str,
        default=os.path.dirname(__file__)
        + f"/../../data/config/subgrapher_config.json",
    )
    parser.add_argument(
        "--images-dir-path",
        type=str,
        default=os.path.dirname(__file__) + f"/../../data/images/default/",
    )
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument(
        "--output-dir-path",
        type=str,
        default=os.path.dirname(__file__) + f"/../../data/predictions/default/",
    )
    parser.add_argument(
        "--output-images-dir-path",
        type=str,
        default=os.path.dirname(__file__) + f"/../../data/visualization/default/",
    )
    args = parser.parse_args()

    # Read SubGrapher config
    with open(args.subgrapher_config_path, "r") as file:
        subgrapher_config = json.load(file)
        subgrapher_config["fg_model_path"] = (
            os.path.dirname(__file__) + f"/../../" + subgrapher_config["fg_model_path"]
        )
        subgrapher_config["cb_model_path"] = (
            os.path.dirname(__file__) + f"/../../" + subgrapher_config["cb_model_path"]
        )
    print("SubGrapher config:", subgrapher_config)

    # Define useful mappings
    fg_substructure_smiles_smarts = pd.read_csv(
        os.path.dirname(__file__) + "/../../data/functional_groups.csv"
    )
    halogen_subtituent_organometallic = pd.read_csv(
        os.path.dirname(__file__) + "/../../data/halogen_subtituent_organometallic.csv"
    )
    fg_mappings = get_mappings(
        fg_substructure_smiles_smarts, halogen_subtituent_organometallic, prefix="fg_"
    )
    cb_substructure_smiles_smarts = pd.read_csv(
        os.path.dirname(__file__) + "/../../data/carbon_chains.csv"
    )
    cb_mappings = get_mappings(
        cb_substructure_smiles_smarts,
        halogen_subtituent_organometallic=None,
        prefix="cb_",
    )
    subgrapher_config.update(fg_mappings)
    subgrapher_config.update(cb_mappings)

    # Define parameters
    image_size = (1024, 1024)
    border_size = 30
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = list(colors.keys())
    white_colors = [
        "beige",
        "bisque",
        "blanchedalmond",
        "cornsilk",
        "antiquewhite",
        "ghostwhite",
        "w",
        "whitesmoke",
        "white",
        "snow",
        "seashell",
        "mistyrose",
        "linen",
        "floralwhite",
        "ivory",
        "honeydew",
        "mintcream",
        "azure",
        "aliceblue",
        "lavenderblush",
    ]
    colors = [color for color in colors if color not in white_colors] * 50
    subgrapher_config.update({"image_size": image_size})

    # Define dataset
    images_paths = [
        filename
        for filename in sorted(glob.glob(args.images_dir_path + "/*"))
        if ((".TIF" in filename) or (".tif" in filename) or (".png" in filename))
    ]
    if images_paths == []:
        print("The provided image directory is empty.")
        exit(0)

    # Remove pre-processed images
    images_paths = [p for p in images_paths if not ("preprocessed" in p)]

    # Define Dataloader (For 32G of GPU memory, the maximum batch size is 55.)
    test_dataset_images = ImagesDatasetInference(
        images_paths,
        image_size,
        border_size=subgrapher_config["image_border_size"],
        binarization_threshold=subgrapher_config["binarization_threshold"],
    )
    test_data_loader_images = DataLoader(
        test_dataset_images,
        shuffle=False,
        batch_size=4,
        num_workers=4,
        collate_fn=test_dataset_images.collate_fn,
    )

    # Load SubGrapher model
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    subgrapher_model = SubGrapher(subgrapher_config, device)

    with torch.no_grad():
        dataiter = iter(test_data_loader_images)
        sample_index = 0
        for i in tqdm(range(len(test_data_loader_images))):
            images_batch, images_paths_batch = next(dataiter)

            fingerprints_batch, matrices_batch, annotations_batch = (
                subgrapher_model.predict_fingerprints(
                    images_batch,
                    output_annotations=args.visualize,
                )
            )

            for batch_index in range(len(images_batch)):
                image, image_path = (
                    images_batch[batch_index],
                    images_paths_batch[batch_index],
                )
                fingerprint, matrix, annotation = (
                    fingerprints_batch[batch_index],
                    matrices_batch[batch_index],
                    annotations_batch[batch_index],
                )
                image_name = image_path.split("/")[-1][:-4]

                if args.visualize:
                    substructures_path = (
                        args.output_images_dir_path
                        + "/"
                        + image_name
                        + "_substructures.png"
                    )
                    caption_path = (
                        args.output_images_dir_path
                        + "/"
                        + image_name
                        + "_substructures.txt"
                    )
                    fingerprint_path = (
                        args.output_images_dir_path
                        + "/"
                        + image_name
                        + "_fingerprint.png"
                    )

                    # Display
                    display_predictions(
                        image,
                        annotation,
                        subgrapher_config,
                        colors,
                        substructures_path,
                        caption_path,
                    )
                    display_fingerprint_matrix(matrix, fingerprint_path)

                # Save
                fingerprint_path = (
                    args.output_dir_path + "/" + image_name + "_fingerprint.npz"
                )
                annotations_path = (
                    args.output_dir_path + "/" + image_name + "_substructure.jsonl"
                )
                with open(annotations_path, "w") as f:
                    for item in annotation:
                        json_line = json.dumps(item)
                        f.write(json_line + "\n")
                sparse.save_npz(fingerprint_path, fingerprint.tocsr())


if __name__ == "__main__":
    main()
