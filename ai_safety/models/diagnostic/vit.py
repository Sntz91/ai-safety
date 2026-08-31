import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,
            in_chans=3,
            drop_path_rate=drop_path_rate
        )
        embed_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.dropout(features)
        cls_token = features[:, 0]
        logits = self.head(cls_token)
        return logits
