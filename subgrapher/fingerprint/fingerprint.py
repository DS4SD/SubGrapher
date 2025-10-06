import math
from collections import defaultdict
from pprint import pprint
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm


def increment_cell(fingerprint, label, factor):
    carbon_chains_labels = range(1534, 1561, 1)
    if label in carbon_chains_labels:
        fingerprint[label] += 1 / factor
    else:
        fingerprint[label] += 2 / factor


def intersects(substructure_a, substructure_b, expansion=0):
    x1_min, y1_min, x1_max, y1_max = substructure_a["bbox"]
    x2_min, y2_min, x2_max, y2_max = substructure_b["bbox"]

    x1_min -= expansion
    y1_min -= expansion
    x1_max += expansion
    y1_max += expansion
    x2_min -= expansion
    y2_min -= expansion
    x2_max += expansion
    y2_max += expansion

    # Check for overlap on x-axis
    x_overlap = not (x1_max < x2_min or x2_max < x1_min)
    # Check for overlap on y-axis
    y_overlap = not (y1_max < y2_min or y2_max < y1_min)

    return x_overlap and y_overlap


def indices_intersects(substructure_a, substructure_b):
    for index in substructure_a["indices"]:
        if index in substructure_b["indices"]:
            return True
    return False


def compute_fingerprint(
    substructures_annotations,
    substructures_labels,
    output_matrix=False,
    use_matches_indices=False,
    expansion_percentage=0.1,
    verbose=False,
):
    label_matches = defaultdict(list)
    for substructure in substructures_labels.keys():
        for detection_id, substructure_annotation in enumerate(
            substructures_annotations
        ):
            if not (substructure_annotation["substructure"] == substructure):
                continue

            label_matches[substructures_labels[substructure]].append(detection_id)

    number_detections = len(substructures_annotations)

    distance_matrix = [
        [np.Inf for _ in range(number_detections)] for _ in range(number_detections)
    ]

    # Find expansion value
    if not (use_matches_indices):
        if len(substructures_annotations) > 0:
            diagonals = [
                math.sqrt(
                    (
                        substructures_annotation["bbox"][2]
                        - substructures_annotation["bbox"][0]
                    )
                    ** 2
                    + (
                        substructures_annotation["bbox"][3]
                        - substructures_annotation["bbox"][1]
                    )
                    ** 2
                )
                for substructures_annotation in substructures_annotations
            ]
            expansion_value = min(diagonals) * expansion_percentage
        else:
            expansion_value = 0

    for detection_id1 in range(number_detections):
        for detection_id2 in range(number_detections):
            if detection_id1 == detection_id2:
                distance_matrix[detection_id1][detection_id2] = 0

            if not (use_matches_indices):
                if intersects(
                    substructures_annotations[detection_id1],
                    substructures_annotations[detection_id2],
                    expansion=expansion_value,
                ):
                    distance_matrix[detection_id1][detection_id2] = 1
            else:
                if indices_intersects(
                    substructures_annotations[detection_id1],
                    substructures_annotations[detection_id2],
                ):
                    distance_matrix[detection_id1][detection_id2] = 1

    # Floyd Warshall Algorithm
    for r in range(number_detections):
        for p in range(number_detections):
            for q in range(number_detections):
                distance_matrix[p][q] = min(
                    distance_matrix[p][q], distance_matrix[p][r] + distance_matrix[r][q]
                )

    matrix = [
        [0 for _ in range(len(substructures_labels))]
        for _ in range(len(substructures_labels))
    ]

    for label, matches in label_matches.items():
        fingerprint = [0 for _ in range(len(substructures_labels))]
        fingerprint[label] = 10 * len(matches)
        for detection_id1 in matches:
            for detection_id2 in range(number_detections):
                if distance_matrix[detection_id1][detection_id2] == 1:
                    increment_cell(
                        fingerprint,
                        substructures_labels[
                            substructures_annotations[detection_id2]["substructure"]
                        ],
                        1,
                    )

                if distance_matrix[detection_id1][detection_id2] == 2:
                    increment_cell(
                        fingerprint,
                        substructures_labels[
                            substructures_annotations[detection_id2]["substructure"]
                        ],
                        4,
                    )

                if distance_matrix[detection_id1][detection_id2] == 3:
                    increment_cell(
                        fingerprint,
                        substructures_labels[
                            substructures_annotations[detection_id2]["substructure"]
                        ],
                        16,
                    )

                if distance_matrix[detection_id1][detection_id2] == 4:
                    increment_cell(
                        fingerprint,
                        substructures_labels[
                            substructures_annotations[detection_id2]["substructure"]
                        ],
                        256,
                    )

        matrix[label] = fingerprint

    if verbose:
        print(
            "The matrix is symmetric: ",
            np.allclose(np.asarray(matrix), np.asarray(matrix).T, atol=1e-8),
        )

    # matrix = np.triu(np.asarray(matrix), k=0)
    triangle_matrix = np.triu(np.asarray(matrix), k=0)  # Keep only upper triangle
    linear_matrix = triangle_matrix.flatten()  # Flatten
    linear_matrix = sparse.coo_matrix(linear_matrix)  # Compress
    linear_matrix = linear_matrix.todok()  # Change format (Dictonary of keys)

    fingerprint = linear_matrix

    if output_matrix:
        return fingerprint, triangle_matrix
    else:
        return fingerprint, None
