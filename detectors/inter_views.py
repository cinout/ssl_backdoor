# import misc
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import json

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


def get_ss_score(full_cov, use_centered_cov=False, debug_print_views=False):
    # full_cov: [bs*n_view, 512]
    """
    https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    """
    full_mean = np.mean(full_cov, axis=0, keepdims=True)
    centered_cov = full_cov - full_mean
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    eigs = v[0:1]  # [1, 512]

    # matmul (n,k),(k,m)->(n,m) on the last two dims
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
        self.seed = args.seed
        self.aug_type = args.aug_type
        self.top_quantile = args.top_quantile
        self.ss_option = args.ss_option

    def forward(self, model, images, gt=None):
        if self.debug_print_views:
            bs = int(images.shape[0] / self.num_views)
            if 1 not in gt:
                return torch.zeros(size=(bs,))

            export_results = dict()
            export_results["seed"] = self.seed
            export_results["aug_type"] = self.aug_type

            # see which one is poisoned image
            print(gt)

            poison_image_index = gt.index(1)
            poison_views_indices = [
                view_i * bs + poison_image_index for view_i in range(self.num_views)
            ]

            unnormalized_images = invTrans(images)
            for i in range(unnormalized_images.shape[0]):
                PIL_image = transforms.functional.to_pil_image(unnormalized_images[i])
                PIL_image.save(f"../visual_{i}.jpg", "JPEG")

            # FIXME: manually record this for poisoned images
            # should ONLY have 1 poisoned image in the batch
            full_trigger_in_view = (
                []
            )  # record the GLOBAL indices of views, full or near-full
            partial_trigger_in_view = (
                []
            )  # as long as part of it is in, even if a very small portion
            no_trigger_in_view = (
                []
            )  # definitely not a single trace of trigger in the view

            # FIXME: for BLEND only
            full_trigger_in_view = poison_views_indices

            all_views = (
                full_trigger_in_view + partial_trigger_in_view + no_trigger_in_view
            )

            # FIXME: toggle
            # exit()

            additional_indices = np.setdiff1d(
                np.array(all_views), np.array(poison_views_indices)
            )
            missing_indices = np.setdiff1d(
                np.array(poison_views_indices), np.array(all_views)
            )

            if len(additional_indices) > 0:
                print(f"unwanted additional indices: {additional_indices}")
            if len(missing_indices) > 0:
                print(f"missing indices: {missing_indices}")

            assert (
                len(all_views) == self.num_views
            ), "the total amount of views is not right"

            export_results["full_trigger_global_indices"] = full_trigger_in_view
            export_results["partial_trigger_global_indices"] = partial_trigger_in_view
            export_results["no_trigger_global_indices"] = no_trigger_in_view

        vision_features = model(images)  # [bs*n_views, 512]
        _, c = vision_features.shape

        if self.debug_print_views and self.interview_task == "spectral_signature":
            export_results["vision_features"] = vision_features.detach().cpu().numpy()
            """
            EXPORT
            """
            with open(f"../SS_AUG_{self.aug_type}_SEED_{self.seed}.npy", "wb") as f:
                np.save(f, export_results)
            exit()

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
                similarity_matrix = vision_features @ vision_features.transpose(
                    1, 2
                )  # [bs, n_view, n_view]
            else:
                raise Exception(f"unimplemented similarity_type {self.similarity_type}")

            # get upper right triangle (off-diagonal) of the matrix
            off_diag_indices = torch.triu_indices(
                row=self.num_views, col=self.num_views, offset=1
            )
            off_diag_indices = off_diag_indices.T

            if self.debug_print_views:
                """
                POISONED
                """
                # get similarity info of poisoned image
                poisoned_image_indice = gt.index(1)
                export_results["poisoned_image_indice"] = poisoned_image_indice
                poisoned_simmatrix = (
                    similarity_matrix[poisoned_image_indice].detach().cpu().numpy()
                )  # [n_view, n_view]

                # convert GLOBAL to LOCAL indices
                full_trigger_in_view = [item // bs for item in full_trigger_in_view]
                partial_trigger_in_view = [
                    item // bs for item in partial_trigger_in_view
                ]
                no_trigger_in_view = [item // bs for item in no_trigger_in_view]

                len_full_trigger_in_view = len(full_trigger_in_view)
                len_partial_trigger_in_view = len(partial_trigger_in_view)
                len_no_trigger_in_view = len(no_trigger_in_view)

                inter_full_trigger_sims = []
                inter_partial_trigger_sims = []
                inter_no_trigger_sims = []

                cross_full_partial_sims = []
                cross_full_no_sims = []
                cross_partial_no_sims = []

                if len_full_trigger_in_view >= 2:
                    for i in range(len_full_trigger_in_view - 1):
                        for j in range(i + 1, len_full_trigger_in_view):
                            inter_full_trigger_sims.append(
                                poisoned_simmatrix[
                                    full_trigger_in_view[i], full_trigger_in_view[j]
                                ]
                            )
                if len_partial_trigger_in_view >= 2:
                    for i in range(len_partial_trigger_in_view - 1):
                        for j in range(i + 1, len_partial_trigger_in_view):
                            inter_partial_trigger_sims.append(
                                poisoned_simmatrix[
                                    partial_trigger_in_view[i],
                                    partial_trigger_in_view[j],
                                ]
                            )
                if len_no_trigger_in_view >= 2:
                    for i in range(len_no_trigger_in_view - 1):
                        for j in range(i + 1, len_no_trigger_in_view):
                            inter_no_trigger_sims.append(
                                poisoned_simmatrix[
                                    no_trigger_in_view[i], no_trigger_in_view[j]
                                ]
                            )

                if len_full_trigger_in_view > 0 and len_partial_trigger_in_view > 0:
                    for item_a in full_trigger_in_view:
                        for item_b in partial_trigger_in_view:
                            cross_full_partial_sims.append(
                                poisoned_simmatrix[item_a, item_b]
                            )

                if len_full_trigger_in_view > 0 and len_no_trigger_in_view > 0:
                    for item_a in full_trigger_in_view:
                        for item_b in no_trigger_in_view:
                            cross_full_no_sims.append(
                                poisoned_simmatrix[item_a, item_b]
                            )
                if len_partial_trigger_in_view > 0 and len_no_trigger_in_view > 0:
                    for item_a in partial_trigger_in_view:
                        for item_b in no_trigger_in_view:
                            cross_partial_no_sims.append(
                                poisoned_simmatrix[item_a, item_b]
                            )

                export_results["inter_full_trigger_sims"] = inter_full_trigger_sims
                export_results["inter_partial_trigger_sims"] = (
                    inter_partial_trigger_sims
                )
                export_results["inter_no_trigger_sims"] = inter_no_trigger_sims
                export_results["cross_full_partial_sims"] = cross_full_partial_sims
                export_results["cross_full_no_sims"] = cross_full_no_sims
                export_results["cross_partial_no_sims"] = cross_partial_no_sims

                """
                CLEAN
                """
                inter_clean_sims = (
                    []
                )  # should contain bs-1 elements, each element is a list of 120 pairs of similiarities

                for i in range(bs):
                    if i == poisoned_image_indice:
                        continue

                    all_sim_values = []  # for this image
                    clean_simmatrix = (
                        similarity_matrix[i].detach().cpu().numpy()
                    )  # [n_view, n_view]
                    for index in off_diag_indices:
                        row, col = index
                        all_sim_values.append(clean_simmatrix[row, col])

                    inter_clean_sims.append(all_sim_values)

                export_results["inter_clean_sims"] = inter_clean_sims

                """
                EXPORT
                """
                with open(f"../VAR_seed{self.seed}.npy", "wb") as f:
                    np.save(f, export_results)

                exit()

            var_of_batch = []
            for i in range(bs):
                # for each image
                sim_matrix_single = similarity_matrix[i]  # [n_views, n_views]
                all_values = []  # all the similarities between views
                for index in off_diag_indices:
                    row, col = index
                    all_values.append(sim_matrix_single[row, col])
                all_values = torch.stack(all_values)

                if self.top_quantile < 1.0:
                    # remove extreme values
                    top_similarity_threshold = torch.quantile(
                        all_values, q=self.top_quantile
                    )
                    all_values = all_values[all_values < top_similarity_threshold]

                var = torch.var(all_values)
                var_of_batch.append(var)

            var_of_batch = torch.stack(var_of_batch)
            return var_of_batch
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
                debug_print_views=self.debug_print_views,
            )
            ss_scores = ss_scores.reshape(self.num_views, -1)  # [n_views, bs]
            if self.ss_option == "mean":
                ss_scores = torch.mean(torch.tensor(ss_scores), dim=0)  # [bs]
            elif self.ss_option == "max":
                ss_scores, _ = torch.max(torch.tensor(ss_scores), dim=0)  # [bs]
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
