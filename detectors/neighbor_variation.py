# import misc
import torch
import torch.nn as nn


class NeighborVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, images):
        vision_features, neighbors = model(images)

        x = torch.zeros([neighbors.shape[0]])

        for i in range(len(neighbors)):
            x[i] = len(neighbors[i].unique())

        return -1 * x
