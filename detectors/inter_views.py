# import misc
import torch
import torch.nn as nn
import numpy as np


def lid_mle(data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    b = data.shape[0]
    k = min(k, b - 2)

    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)

    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.0e-4), dim=1)
    return lids


def get_ss_score(full_cov, use_centered_cov=False):
    """
    https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    """
    full_mean = np.mean(full_cov, axis=0, keepdims=True)
    centered_cov = full_cov - full_mean
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    eigs = v[0:1]
    if use_centered_cov:
        corrs = np.matmul(eigs, np.transpose(centered_cov))
    else:
        corrs = np.matmul(eigs, np.transpose(full_cov))
    scores = np.linalg.norm(corrs, axis=0)  # 2-norm by default
    return scores


class InterViews(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, images, args):
        vision_features = model(images)  # [bs*n_views, 512]
        _, c = vision_features.shape

        if args.interview_task == "variance":
            vision_features = vision_features.reshape(
                args.num_views, -1, c
            )  # [n_views, bs, 512]
            vision_features = torch.permute(
                vision_features, (1, 0, 2)
            )  # [bs, n_views, 512]
            bs = vision_features.shape[0]

            if args.similarity_type == "cosine":
                vision_features = (
                    vision_features / vision_features.norm(dim=2)[:, :, None]
                )
                similarity_matrix = vision_features @ vision_features.transpose(
                    1, 2
                )  # [bs, n_view, n_view], diagonals are 1.00
            elif args.similarity_type == "raw":
                similarity_matrix = vision_features @ vision_features.transpose(1, 2)
            else:
                raise Exception(f"unimplemented similarity_type {args.similarity_type}")

            # get upper right triangle (off-diagonal) of the matrix
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
        elif args.interview_task == "lid":
            lids = lid_mle(data=vision_features, reference=vision_features)
            lids = lids.reshape(args.num_views, -1)  # [n_views, bs]
            lids = torch.mean(lids, dim=0)  # [bs]
            return lids
        elif args.interview_task == "entropy":
            pass
        elif args.interview_task == "spectral_signature":
            ss_scores = get_ss_score(
                vision_features.detach().cpu().numpy(),
                use_centered_cov=args.use_centered_cov,
            )
            ss_scores = ss_scores.reshape(args.num_views, -1)  # [n_views, bs]
            ss_scores = torch.mean(torch.tensor(ss_scores), dim=0)  # [bs]
            return ss_scores
        elif args.interview_task == "effective_rank":
            pass
        else:
            raise Exception(
                f"this interview_task {args.interview_task} is unimplemented yet."
            )
