# import misc
import torch
import torch.nn as nn


class InterViews(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, images, args):
        vision_features = model(images)  # [bs*n_views, 512]
        _, c = vision_features.shape
        vision_features = vision_features.reshape(
            args.num_views, -1, c
        )  # [n_views, bs, 512]
        vision_features = torch.permute(
            vision_features, (1, 0, 2)
        )  # [bs, n_views, 512]

        # TODO: update all tasks
        if args.interview_task == "variance":
            vision_features = vision_features / vision_features.norm(dim=2)[:, :, None]
            similarity_matrix = vision_features @ vision_features.transpose(
                1, 2
            )  # [bs, n_view, n_view], diagonals are 1.00

        else:
            raise Exception(
                f"this interview_task {args.interview_task} is unimplemented yet."
            )
        return
