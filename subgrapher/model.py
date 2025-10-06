#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from subgrapher.fingerprint.fingerprint import compute_fingerprint
from subgrapher.postprocessing import apply_threshold, revise_predicted_labels


class InstanceSegmentationModel:
    def __init__(self, nb_classes: int):
        self.model = self.get_mask_rcnn(nb_classes)
        return

    def get_mask_rcnn(self, nb_classes: int):
        """
        Returns a pretrained Mask R-CNN model.

        torchvision.models.detection.maskrcnn_resnet50_fpn parameters:
            - pretrained (bool): If True, returns a model pre-trained on COCO train2017
            - progress (bool): If True, displays a progress bar of the download to stderr
            - num_classes (int): number of output classes of the model (including the background)
            - pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
            - trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            - min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
            - max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
            - image_mean (Tuple[float, float, float]): mean values used for input normalization.
               They are generally the mean values of the dataset on which the backbone has been trained on
            - image_std (Tuple[float, float, float]): std values used for input normalization.
               They are generally the std values of the dataset on which the backbone has been trained on
            - box_nms_thresh (float): NMS threshold for the prediction head. Used during inference (Max overlap between predictions of the same class)
            - rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals (Max overlap between boxes, no matter their class. Independent on all feature map. Proposals from different feature maps can overlap more than rpn_nms_thresh)
            - box_detections_per_img (int): maximum number of detections per image, for all classes.
            - rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
            - rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
            - rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
            - rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
            - rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
               for computing the loss
            - box_batch_size_per_image (int): number of proposals that are sampled during training of the
               classification head

        Notes:
            The top 100 scoring boxes are passed to the mask head.
        """

        # Load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            progress=True,
            trainable_backbone_layers=5,
            min_size=1024,
            max_size=1024,
            rpn_nms_thresh=0.90,
            box_nms_thresh=0.50,
            box_detections_per_img=2500,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )  # , box_score_thresh=0, rpn_score_thresh=0, #rpn_nms_thresh=0.95

        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=nb_classes
        )

        # Get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Set output dimension of the first convolution (number of channels)
        hidden_layer = 256

        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=hidden_layer,
            num_classes=nb_classes,
        )

        return model


class SubGrapher:
    """Class for SubGrapher inference."""

    def __init__(self, config, device):
        # Default configuration
        self.config = {
            "fg_nb_classes": 1535,
            "cb_nb_classes": 28,
            "fg_model_path": "./data/models/experiment-052_run-003_epoch-13.pt", 
            "cb_model_path": "./data/models/experiment-051_run-001_epoch-30.pt", 
            "fg_score_threshold": 0.91,
            "cb_score_threshold": 0.92,
            "image_border_size": 50,
            "binarization_threshold": 0.555,
            "image_size": [1024, 1024],
        }

        self.config.update(config)

        self.device = device

        # Carbon-backbone model
        if self.config["cb_model_path"] != "":
            self.cb_model = InstanceSegmentationModel(
                nb_classes=config["cb_nb_classes"]
            ).model.to(self.device)
            if not (os.path.exists(self.config["cb_model_path"])):
                print(
                    f"Downloading the carbon backbone model: {self.config['cb_model_path']}."
                )
                cb_model_name = self.config["cb_model_path"].split("/")[-1]
                subprocess.run(
                    [
                        "wget",
                        f"https://huggingface.co/ds4sd/SubGrapher/resolve/main/models/{cb_model_name}",
                        "-P",
                        f"./data/models/",
                    ],
                    check=True,
                )

            self.cb_model.load_state_dict(
                torch.load(self.config["cb_model_path"], map_location=self.device)[
                    "state_dict"
                ]
            )
            self.cb_model.eval()

        # Functional-group model
        if self.config["fg_nb_classes"] != "":
            self.fg_model = InstanceSegmentationModel(
                nb_classes=config["fg_nb_classes"]
            ).model.to(self.device)
            if not (os.path.exists(self.config["fg_model_path"])):
                print(
                    f"Downloading the functional group model: {self.config['fg_model_path']}."
                )
                fg_model_name = self.config["fg_model_path"].split("/")[-1]
                subprocess.run(
                    [
                        "wget",
                        f"https://huggingface.co/ds4sd/SubGrapher/resolve/main/models/{fg_model_name}",
                        "-P",
                        f"./data/models/",
                    ],
                    check=True,
                )

            self.fg_model.load_state_dict(
                torch.load(self.config["fg_model_path"], map_location=self.device)[
                    "state_dict"
                ]
            )
            self.fg_model.eval()

    def predict_functional_groups(self, images):
        predictions_batch = []
        predictions_batch_filtered = []

        images = list(image.to(self.device) for image in images)
        with torch.no_grad():
            predictions = self.fg_model(images)
        images = list(image.cpu() for image in images)
        predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]

        for batch_index in range(len(images)):
            predictions_batch.append(predictions[batch_index])
            apply_threshold(predictions[batch_index], self.config["fg_score_threshold"])
            predictions_batch_filtered.append(predictions[batch_index])

        return predictions_batch, predictions_batch_filtered

    def predict_carbon_backbones(self, images):
        predictions_batch = []
        predictions_batch_filtered = []

        images = list(image.to(self.device) for image in images)
        with torch.no_grad():
            predictions = self.cb_model(images)
        images = list(image.cpu() for image in images)
        predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]

        for batch_index in range(len(images)):
            predictions_batch.append(predictions[batch_index])
            apply_threshold(predictions[batch_index], self.config["cb_score_threshold"])
            revise_predicted_labels(
                predictions[batch_index],
                self.config["cb_substructures_labels"],
                self.config["image_size"],
            )
            predictions_batch_filtered.append(predictions[batch_index])

        return predictions_batch, predictions_batch_filtered

    def predict_substructures(self, images, return_list=True):
        _, fg_predictions_batch = self.predict_functional_groups(images)
        _, cb_predictions_batch = self.predict_carbon_backbones(images)

        if return_list:
            annotations_batch = [[] for _ in range(len(images))]
            for batch_idx, prediction in enumerate(fg_predictions_batch):
                for index in range(prediction["boxes"].size()[0]):
                    annotations_batch[batch_idx].append(
                        {
                            "substructure": self.config["fg_labels_substructures"][
                                prediction["labels"][index].item()
                            ],
                            "confidence": round(prediction["scores"][index].item(), 4),
                            "bbox": prediction["boxes"][index].tolist(),
                            "type": "functional-groups",
                        }
                    )
            for batch_idx, prediction in enumerate(cb_predictions_batch):
                for index in range(prediction["boxes"].size()[0]):
                    annotations_batch[batch_idx].append(
                        {
                            "substructure": self.config["cb_labels_substructures"][
                                prediction["labels"][index].item()
                            ],
                            "confidence": round(prediction["scores"][index].item(), 4),
                            "bbox": prediction["boxes"][index].tolist(),
                            "type": "carbons-chains",
                        }
                    )
        return annotations_batch

    def predict_fingerprints(self, images, output_annotations=False):
        annotations_batch = self.predict_substructures(images)

        fingerprints = []
        matrices = []
        for batch_index, annotations in enumerate(annotations_batch):
            # Combine substructures
            substructures_labels = {}
            for substructure, index in self.config["fg_substructures_labels"].items():
                substructures_labels[substructure] = index
            for substructure, index in self.config["cb_substructures_labels"].items():
                substructures_labels[substructure] = index + len(
                    self.config["fg_substructures_labels"]
                )

            # Compute fingerprint
            fingerprint, matrix = compute_fingerprint(
                annotations, substructures_labels, output_matrix=output_annotations
            )
            fingerprints.append(fingerprint)
            matrices.append(matrix)

        if output_annotations:
            return fingerprints, matrices, annotations_batch
        else:
            return fingerprints
