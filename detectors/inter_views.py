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
            similarity_matrix = vision_features @ torch.permute(
                vision_features, (0, 2, 1)
            )  # [bs, n_views, n_views]
            print(similarity_matrix)
            exit()

            pass
        else:
            raise Exception(
                f"this interview_task {args.interview_task} is unimplemented yet."
            )
        return


def find_nearest_neighbor(features, topk):
    f2compare = torch.mean(features, dim=(2, 3))
    similarity = f2compare @ f2compare.T
    _, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)
    return indices  # [bs*n_views, topk]
