# import misc
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


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


def effective_rank(z):
    z = np.transpose(z)
    c = np.cov(z)  # convariance matrix
    u, s, vh = np.linalg.svd(c)
    p = s / np.abs(s).sum()
    h = -np.sum(p * np.log(p))
    er = np.exp(h)
    return er


class InterViews(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.debug_print_views = args.debug_print_views
        self.interview_task = args.interview_task
        self.num_views = args.num_views
        self.similarity_type = args.similarity_type
        self.use_centered_cov = args.use_centered_cov

    def forward(self, model, images):
        if self.debug_print_views:
            # TODO: remove [0] and save all
            unnormalized_images = invTrans(images)

            PIL_image = transforms.functional.to_pil_image(unnormalized_images[10])
            PIL_image.save(f"test_random.jpg", "JPEG")
            exit()

        vision_features = model(images)  # [bs*n_views, 512]
        _, c = vision_features.shape

        if self.interview_task == "variance":
            vision_features = vision_features.reshape(
                self.num_views, -1, c
            )  # [n_views, bs, 512]
            vision_features = torch.permute(
                vision_features, (1, 0, 2)
            )  # [bs, n_views, 512]
            bs = vision_features.shape[0]

            if self.similarity_type == "cosine":
                vision_features = (
                    vision_features / vision_features.norm(dim=2)[:, :, None]
                )
                similarity_matrix = vision_features @ vision_features.transpose(
                    1, 2
                )  # [bs, n_view, n_view], diagonals are 1.00
            elif self.similarity_type == "raw":
                similarity_matrix = vision_features @ vision_features.transpose(1, 2)
            else:
                raise Exception(f"unimplemented similarity_type {self.similarity_type}")

            # get upper right triangle (off-diagonal) of the matrix
            off_diag_indices = torch.triu_indices(
                row=self.num_views, col=self.num_views, offset=1
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
        elif self.interview_task == "lid":
            lids = lid_mle(data=vision_features, reference=vision_features)
            lids = lids.reshape(self.num_views, -1)  # [n_views, bs]
            lids = torch.mean(lids, dim=0)  # [bs]
            return lids
        elif self.interview_task == "entropy":
            pass
        elif self.interview_task == "spectral_signature":
            ss_scores = get_ss_score(
                vision_features.detach().cpu().numpy(),
                use_centered_cov=self.use_centered_cov,
            )
            ss_scores = ss_scores.reshape(self.num_views, -1)  # [n_views, bs]
            ss_scores = torch.mean(torch.tensor(ss_scores), dim=0)  # [bs]
            return ss_scores
        elif self.interview_task == "effective_rank":
            vision_features = vision_features.reshape(
                self.num_views, -1, c
            )  # [n_views, bs, 512]
            vision_features = torch.permute(
                vision_features, (1, 0, 2)
            )  # [bs, n_views, 512]
            bs = vision_features.shape[0]

            erank_of_batch = []
            for i in range(bs):
                features_of_n_views = vision_features[i]  # [n_views, 512]
                eff_rank = effective_rank(
                    features_of_n_views.detach().cpu().numpy(),
                )  # score
                erank_of_batch.append(torch.tensor(eff_rank))

            erank_of_batch = torch.stack(erank_of_batch)
            return -erank_of_batch
        else:
            raise Exception(
                f"this interview_task {self.interview_task} is unimplemented yet."
            )
