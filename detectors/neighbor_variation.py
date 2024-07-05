# import misc
import torch
import torch.nn as nn


# TODO: update
class NeighborVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, images):
        vision_features, neighbors = model(images)  # shape: [bs, 512]

        _, counts = torch.unique(neighbors, return_counts=True)

        return -1 * counts
