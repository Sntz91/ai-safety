import torch.nn as nn
import timm


class DenseNet(nn.Module):
    def __init__(self, dropout=0.5, drop_rate=0.1, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            'densenet169', pretrained=True, global_pool='avg', num_classes=0,
            drop_rate=drop_rate,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1664, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
