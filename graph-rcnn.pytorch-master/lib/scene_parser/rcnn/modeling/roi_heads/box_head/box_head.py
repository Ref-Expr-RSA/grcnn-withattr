# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from .attr_predictor import attr_predictor
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from lib.scene_parser.rcnn.modeling.utils import cat

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.attr_predictor=attr_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training: # or not self.cfg.inference:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        # print(len(features))
        # print(type(features[0]),type(features))
        # print(features[0].shape)
        # print(type(proposals),type(targets))
        x = self.feature_extractor(features, proposals)
        # print(proposals[0].get_field('labels'))
        # print(len(proposals))
        # print(len(proposals[0]),len(proposals[1]))
        # print(type(x))
        # print(x.shape)
        # print(len(x))
        # final classifier that converts the features into predictions
        class_logits, box_regression,  = self.predictor(x)
        label_value = None
        if not self.training:
            label_value= torch.argmax(class_logits, dim=1)
            # print(label_value)
        else:
            label_value=cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        # print(label_value.shape)
        attr_logits=self.attr_predictor(x,label_value)

        # class_logits, box_regression = self.predictor(x)

        boxes_per_image = [len(proposal) for proposal in proposals]
        features = x.split(boxes_per_image, dim=0)
        for proposal, feature in zip(proposals, features):
            proposal.add_field("features", self.avgpool(feature))
        if not self.training:
            # if self.cfg.inference:
            # result = self.post_processor((class_logits, box_regression), proposals)
            result = self.post_processor((class_logits, box_regression,attr_logits), proposals)
            # print('test....')
            if targets:
                # print('if targets...')
                result = self.loss_evaluator.prepare_labels(result, targets)
            return x, result, {}
            # else:
                # return x, proposals, {}

        loss_classifier, loss_box_reg,attr_classifer = self.loss_evaluator(
            [class_logits], [box_regression],[attr_logits]
        )
        class_logits = class_logits.split(boxes_per_image, dim=0)
        # attr_logits = attr_logits.split(boxes_per_image, dim=0)
        for proposal, class_logit in zip(proposals, class_logits):
            proposal.add_field("logits", class_logit)
            # proposal.add_field("attr_logits", attr_logits)

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg,attr_classifer=attr_classifer),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
