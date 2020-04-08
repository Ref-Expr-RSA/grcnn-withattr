from torch import nn
import torch
class attributePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(attributePredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels
        num_attr=config.NUM_ATTR

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs+num_classes, num_attr)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    # def forward(self, x):
    #     cl = self.avgpool(x)
    #     cl = cl.view(cl.size(0), -1)
    #     cls_logit = self.cls_score(cl)
    #     return cls_logit

    def forward(self, x, label):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        label = label.cpu().reshape(-1, 1)
        label = torch.zeros(len(label), 151).scatter_(1, label, 1).to(x.device)
        x_concat = torch.cat((x, label), 1)
        attr_pred = self.cls_score(x_concat)
        return attr_pred
    def make_roi_relation_box_predictor(cfg, in_channels):
        return attributePredictor(cfg, in_channels)