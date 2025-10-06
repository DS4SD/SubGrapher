#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

# Standard
import os
import sys
from collections import defaultdict

# Image
import cv2
import numpy as np
import PIL

# Mathematics
import torch
from PIL import Image, ImageOps

# Chemistry
from rdkit import Chem


def get_molecule_from_smiles(smiles, remove_stereochemistry):
    # sanitize = False is mandatory for H abreviations
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    if molecule is None:
        return None

    if remove_stereochemistry:
        Chem.RemoveStereochemistry(molecule)

    molecule.UpdatePropertyCache(strict=False)
    sanity = Chem.SanitizeMol(
        molecule,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )

    if sanity != Chem.rdmolops.SANITIZE_NONE:
        return None

    return molecule


def get_molecule_from_molfile(molfile, remove_stereochemistry):
    molecule = Chem.MolFromMolFile(molfile, sanitize=False)
    if molecule is None:
        return None

    if remove_stereochemistry:
        Chem.RemoveStereochemistry(molecule)

    molecule.UpdatePropertyCache(strict=False)
    sanity = Chem.SanitizeMol(
        molecule,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )

    if sanity != Chem.rdmolops.SANITIZE_NONE:
        return None

    return molecule


def iou(
    box_target: torch.Tensor,
    boxes_prediction: torch.Tensor,
    area_target: torch.Tensor,
    areas_prediction: torch.Tensor,
):
    """
    Computes the intersection over union between one box and a set of boxes.
    The areas are passed in rather than calculated here for efficiency.
    Calculate once in the caller to avoid duplicate work.
    """
    y_max = np.maximum(box_target[0], boxes_prediction[:, 0])
    y_min = np.minimum(box_target[2], boxes_prediction[:, 2])
    x_max = np.maximum(box_target[1], boxes_prediction[:, 1])
    x_min = np.minimum(box_target[3], boxes_prediction[:, 3])
    intersection = np.maximum(x_min - x_max, 0) * np.maximum(y_min - y_max, 0)
    union = area_target + areas_prediction[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes_prediction: torch.Tensor, boxes_target: torch.Tensor):
    """
    Computes intersection over union between 2 set of boxes.
    For better performance, pass the largest set first and the smaller second.
    """
    # Computes areas of anchors and ground truth boxes
    areas_prediction = (boxes_prediction[:, 2] - boxes_prediction[:, 0]) * (
        boxes_prediction[:, 3] - boxes_prediction[:, 1]
    )
    areas_target = (boxes_target[:, 2] - boxes_target[:, 0]) * (
        boxes_target[:, 3] - boxes_target[:, 1]
    )

    # Compute overlaps
    overlaps = np.zeros((boxes_prediction.shape[0], boxes_target.shape[0]))
    for i in range(overlaps.shape[1]):
        box_target = boxes_target[i]
        overlaps[:, i] = iou(
            box_target, boxes_prediction, areas_target[i], areas_prediction
        )
    return overlaps


def resize_image(image, image_size: tuple, border_size: int = 30, b="white"):
    """
    Transforms the image, by applying :
        - A resizing, keeping the image scale ratio
        - A padding to create a square image
        - An additionnal padding on borders
    Returns:
        - 1-channel image.
    """
    width = image_size[0] - 2 * border_size
    height = image_size[1] - 2 * border_size
    width_ratio = width / image.width
    height_ratio = height / image.height

    if width_ratio < height_ratio:
        transformed_width = width
        transformed_height = round(width_ratio * image.height)
    else:
        transformed_width = round(height_ratio * image.width)
        transformed_height = height

    # Rescale and pad
    transformed_image = image.resize(
        (transformed_width, transformed_height), Image.LANCZOS
    )
    if b == "black":
        background = Image.new("L", (width, height), (0))
    if b == "white":
        background = Image.new("L", (width, height), (255))
    offset = (
        round((width - transformed_width) / 2),
        round((height - transformed_height) / 2),
    )
    background.paste(transformed_image, offset)

    # Add a padding on image borders
    if b == "black":
        transformed_image = ImageOps.expand(
            background, border=border_size, fill="black"
        )
    if b == "white":
        transformed_image = ImageOps.expand(
            background, border=border_size, fill="white"
        )

    return transformed_image


def crop_tight(pil_image, dilate=False):
    if dilate:
        # Dilate image to remove isolated pixels
        im = np.array(pil_image, dtype=np.uint8)
        im = cv2.dilate(
            im, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
        )
        im = Image.fromarray(im)
    else:
        im = pil_image.copy()

    bbox = PIL.ImageOps.invert(im).getbbox()
    if bbox is None:
        return pil_image

    min_cropping_size = 60
    if (abs(bbox[2] - bbox[0]) > min_cropping_size) or (
        abs(bbox[3] - bbox[1]) > min_cropping_size
    ):
        return pil_image.crop(bbox)

    elif dilate:
        print("Recursive borders cropping")
        return crop_tight(pil_image, dilate=False)

    else:
        print("Fixed window cropping")
        missing_x = min_cropping_size - abs(bbox[2] - bbox[0])
        missing_y = min_cropping_size - abs(bbox[3] - bbox[1])
        print(bbox)
        bbox = [
            bbox[0] - missing_x // 2,
            bbox[1] - missing_y // 2,
            bbox[2] + missing_x // 2,
            bbox[3] + missing_y // 2,
        ]
        return pil_image.crop(bbox)


def get_substructure_matches(
    smiles="",
    molecule=None,
    subsmarts_labels={},
    substructures_subsmarts={},
    banned_indices=[],
):
    if molecule is None:
        molecule = get_molecule_from_smiles(
            smiles, remove_stereochemistry=False, remove_hs=True
        )

    # Add Hs to match patterns with Hs. This is not shifting matches indices because H atoms are always labelled with the last indices.
    molecule = Chem.AddHs(molecule)

    image_matches = {}

    for subsmarts in subsmarts_labels.keys():

        submolecule = Chem.MolFromSmarts(subsmarts)
        matches = molecule.GetSubstructMatches(submolecule)

        # Remove Hs and groups containing R labels from matches (Tricky logic but works)
        matches_list = [list(match) for match in matches]
        for match_index, match in enumerate(matches_list):
            remove_indices_atoms = []
            banned = False
            for atom_index in match:
                # Remove groups containing R labels
                if atom_index in banned_indices:
                    matches_list[match_index] = []
                    banned = True
                    break
                # Remove H
                if molecule.GetAtomWithIdx(atom_index).GetSymbol() == "H":
                    remove_indices_atoms.append(atom_index)
            if not banned:
                matches_list[match_index] = [
                    index
                    for index in matches_list[match_index]
                    if index not in remove_indices_atoms
                ]

        matches_list = [match for match in matches_list if match != []]

        if matches_list == []:
            matches = ()
        else:
            matches = tuple([tuple(match) for match in matches_list])

        if matches != ():
            image_matches[subsmarts_labels[subsmarts]] = matches

    # Remove redundant annotations
    if "CC([H])([H])[H]" in substructures_subsmarts.keys():
        for label_sub in [
            subsmarts_labels[substructures_subsmarts[substructure]]
            for substructure in ["CCC", "C=CC", "C=C=C", "C#CC", "CC(C)C", "C=C(C)C"]
        ]:
            if label_sub in image_matches.keys():
                matches_sub = image_matches[label_sub]
                remove_indices_overlaps = []
                for remove_index_overlaps, match_sub in enumerate(matches_sub):
                    removed = False
                    for label_super, matches_super in image_matches.items():
                        if label_super != label_sub:
                            for match_super in matches_super:
                                if set(match_sub).issubset(set(match_super)):
                                    if len(match_sub) < len(match_super):
                                        remove_indices_overlaps.append(
                                            remove_index_overlaps
                                        )
                                        removed = True
                                        break
                                    elif (len(match_sub) == len(match_super)) and (
                                        label_super
                                        in [
                                            subsmarts_labels[
                                                substructures_subsmarts[substructure]
                                            ]
                                            for substructure in [
                                                "Cyclopropane",
                                                "Cyclopropene",
                                                "Cyclopropyne",
                                            ]
                                        ]
                                    ):
                                        remove_indices_overlaps.append(
                                            remove_index_overlaps
                                        )
                                        removed = True
                                        break
                            if removed:
                                break
                image_matches[label_sub] = [
                    match
                    for index, match in enumerate(image_matches[label_sub])
                    if index not in remove_indices_overlaps
                ]

        # Remove empty matches
        image_matches_copy = image_matches.copy()
        for label, matches in image_matches_copy.items():
            if matches == []:
                image_matches.pop(label, None)
    return image_matches


def get_mappings(
    substructure_smiles_smarts, halogen_subtituent_organometallic=None, prefix=""
):
    mappings = {}
    mappings[prefix + "subsmarts_labels"] = {
        k: v
        for k, v in zip(
            substructure_smiles_smarts["SMARTS"],
            range(1, len(substructure_smiles_smarts) + 1),
        )
    }
    mappings[prefix + "substructures_subsmarts"] = {
        k: v
        for k, v in zip(
            substructure_smiles_smarts["Substructure"],
            substructure_smiles_smarts["SMARTS"],
        )
    }
    mappings[prefix + "labels_substructures"] = {
        k: v
        for k, v in zip(
            range(1, len(substructure_smiles_smarts) + 1),
            substructure_smiles_smarts["Substructure"],
        )
    }
    mappings[prefix + "labels_subsmarts"] = {
        k: v
        for k, v in zip(
            range(1, len(substructure_smiles_smarts) + 1),
            substructure_smiles_smarts["SMARTS"],
        )
    }
    mappings[prefix + "substructures_labels"] = {
        k: v
        for k, v in zip(
            substructure_smiles_smarts["Substructure"],
            range(1, len(substructure_smiles_smarts) + 1),
        )
    }
    mappings[prefix + "labels_bonds"] = defaultdict(list)
    for label in mappings[prefix + "subsmarts_labels"].values():
        molecule = Chem.MolFromSmarts(mappings[prefix + "labels_subsmarts"][label])
        molecule = Chem.RemoveAllHs(molecule)
        for bond in molecule.GetBonds():
            mappings[prefix + "labels_bonds"][label].append(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            )
            mappings[prefix + "labels_bonds"][label].append(
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            )

    if halogen_subtituent_organometallic is None:
        return mappings

    mappings[prefix + "halogen_subtituent_organometallic_labels"] = {
        substructure: mappings[prefix + "substructures_labels"][substructure]
        for substructure in halogen_subtituent_organometallic["Substructure"]
    }
    mappings[prefix + "labels_terminal_carbons"] = {}
    for label in mappings[prefix + "subsmarts_labels"].values():
        molecule = Chem.MolFromSmarts(mappings[prefix + "labels_subsmarts"][label])
        Chem.SanitizeMol(molecule)
        matches = molecule.GetSubstructMatches(Chem.MolFromSmarts("[CH3]"))
        if matches != ():
            mappings[prefix + "labels_terminal_carbons"][label] = [
                index for match in matches for index in match
            ]
        else:
            mappings[prefix + "labels_terminal_carbons"][label] = []
    return mappings
