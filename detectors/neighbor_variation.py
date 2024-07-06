# import misc
import torch
import torch.nn as nn


class NeighborVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, images, args, gt):
        vision_features, neighbors = model(images)

        total_bs = neighbors.shape[0]
        actual_bs = total_bs / args.num_views

        # TODO: remove later
        nn_from_same_image = torch.zeros(
            size=(neighbors.shape[0], neighbors.shape[0]), dtype=torch.bool
        )
        print(">>>> nn_from_same_image")
        for i in range(len(neighbors)):
            nn_from_same_image[i] = neighbors[i, :] % actual_bs == i % actual_bs

            print(
                f"GT: {gt[int(i % actual_bs)]}; ",
                nn_from_same_image[i].to(torch.uint8).detach().cpu().numpy(),
            )

        x = torch.zeros([neighbors.shape[0]])

        for i in range(len(neighbors)):
            x[i] = len(neighbors[i].unique())

        return -1 * x
