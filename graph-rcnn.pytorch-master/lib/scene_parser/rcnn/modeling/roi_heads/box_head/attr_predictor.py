from torch import nn
import torch
class attr_predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(attr_predictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels
        num_attrs=config.NUM_ATTR
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attr_score = nn.Linear(num_inputs+num_classes, num_attrs)

        nn.init.normal_(self.attr_score.weight, mean=0, std=0.001)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x,label):
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        label=label.cpu().reshape(-1,1)
        label=torch.zeros(len(label), 151).scatter_(1, label, 1).to(x.device)
        x_concat=torch.cat((x,label),1)
        attr_pred = self.attr_score(x_concat)
        return attr_pred