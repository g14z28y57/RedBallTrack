import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


class VisionBackbone(nn.Module):
    def __init__(self, out_layer):
        super().__init__()
        vision_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(vision_backbone, return_nodes={out_layer: "out"}).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        return self.feature_extractor(x)["out"]


class VisionEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        return x


class ResBlock(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.BatchNorm1d(d_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(d_feedforward, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        out = x + out
        out = nn.ReLU()(out)
        return out


class FeatureFuser(nn.Module):
    def __init__(self, d_input, d_model, d_feedforward, num_layers, d_direction=3, d_distance=1, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([ResBlock(d_model, d_feedforward, dropout=dropout) for _ in range(num_layers)])
        self.direction_head = nn.Linear(d_model, d_direction)
        self.distance_head = nn.Linear(d_model, d_distance)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        direction = self.direction_head(x)
        distance = self.distance_head(x)
        direction = nn.Hardtanh()(direction)
        distance = nn.ReLU()(distance)
        return direction, distance


class DirectionModel(nn.Module):
    def __init__(self, d_input, d_model, d_feedforward, out_channels, num_layers, d_direction=3, d_distance=1, dropout=0.3):
        super().__init__()
        self.vision = VisionEncoder(out_channels)
        self.fuser = FeatureFuser(d_input=d_input, d_model=d_model, d_feedforward=d_feedforward, num_layers=num_layers,
                                  d_direction=d_direction, d_distance=d_distance, dropout=dropout)

    def forward(self, img, camera_pos, camera_front):
        img_feature = self.vision(img)
        feature = torch.concatenate([img_feature, camera_pos, camera_front], dim=-1)
        direction, distance = self.fuser(feature)
        return direction, distance
