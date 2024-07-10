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

        bs = vision_features.shape[0]

        # TODO: update all tasks
        if args.interview_task == "variance":
            vision_features = vision_features / vision_features.norm(dim=2)[:, :, None]
            similarity_matrix = vision_features @ vision_features.transpose(
                1, 2
            )  # [bs, n_view, n_view], diagonals are 1.00
            off_diag_indices = torch.triu_indices(
                row=args.num_views, col=args.num_views, offset=1
            )
            off_diag_indices = off_diag_indices.T

            var_of_batch = []
            for i in range(bs):
                sim_matrix_single = similarity_matrix[i]  # [n_views, n_views]
                all_values = []
                for index in off_diag_indices:
                    row, col = index
                    all_values.append(sim_matrix_single[row, col])
                all_values = torch.stack(all_values)
                var = torch.var(all_values)
                var_of_batch.append(var)

            var_of_batch = torch.stack(var_of_batch)
            return -1 * var_of_batch

        else:
            raise Exception(
                f"this interview_task {args.interview_task} is unimplemented yet."
            )
