import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moco.moco.loader import NCropsTransform
from moco.moco.loader import GaussianBlur
import random
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score


seeds = [30, 42]
augs = ["basic_plus_rotation_rigid", "crop_plus_perspective", "perspective"]

num_views = 32
batch_size = 128
gt = np.zeros(batch_size, dtype=np.uint)
gt[0] = 1

for aug in augs:
    for seed in seeds:
        with open(f"SS_AUG_{aug}_SEED_{seed}.npy", "rb") as f:
            results = np.load(f, allow_pickle=True)  # a numpy array, not a dict
            results = results[()]  # a dict

            full_trigger_global_indices = results["full_trigger_global_indices"]
            partial_trigger_global_indices = results["partial_trigger_global_indices"]
            no_trigger_global_indices = results["no_trigger_global_indices"]
            # print(partial_trigger_global_indices, no_trigger_global_indices)
            # others are all clean image's views

            vision_features = results["vision_features"]
            bs, channel = vision_features.shape

            #### [analysis: data preparation]
            if len(full_trigger_global_indices) > 0:
                full_trigger_features = vision_features[
                    np.array(full_trigger_global_indices)
                ]
                full_trigger_features_mean = np.mean(full_trigger_features, axis=0)
            else:
                full_trigger_features_mean = None

            if len(partial_trigger_global_indices) > 0:
                partial_trigger_features = vision_features[
                    np.array(partial_trigger_global_indices)
                ]
                partial_trigger_features_mean = np.mean(
                    partial_trigger_features, axis=0
                )
            else:
                partial_trigger_features_mean = None

            if len(no_trigger_global_indices) > 0:
                no_trigger_features = vision_features[
                    np.array(no_trigger_global_indices)
                ]
                no_trigger_features_mean = np.mean(no_trigger_features, axis=0)
            else:
                no_trigger_features_mean = None

            clean_indices = list(
                set(range(bs))
                - set(
                    full_trigger_global_indices
                    + partial_trigger_global_indices
                    + no_trigger_global_indices
                )
            )
            clean_features = vision_features[np.array(clean_indices)]
            clean_features_mean = np.mean(clean_features, axis=0)  # [512]
            clean_features_std = np.std(clean_features, axis=0)  # [512]

            vision_features_mean = np.mean(vision_features, axis=0)
            #### [END OF analysis: data preparation]

            # #### [ANALYSIS: shift in mean]
            shift_in_mean_l1 = np.linalg.norm(
                vision_features_mean - clean_features_mean, ord=1
            )
            shift_in_mean_l2 = np.linalg.norm(
                vision_features_mean - clean_features_mean
            )
            print(f"shift_in_mean L1: {shift_in_mean_l1}")
            print(f"shift_in_mean L2: {shift_in_mean_l2}")
            # #### [END OF ANALYSIS: shift in mean]

            # ### [ANALYSIS: in bound or not?]
            # lower_bound_1std = clean_features_mean - clean_features_std
            # upper_bound_1std = clean_features_mean + clean_features_std

            # in_1std = 0
            # out_1std = 0
            # for full, lower, upper in zip(
            #     full_trigger_features_mean, lower_bound_1std, upper_bound_1std
            # ):
            #     if full >= lower and full <= upper:
            #         in_1std += 1
            #     else:
            #         out_1std += 1
            # print(
            #     f"Full_trigger || in_1std: {in_1std}, out_1std: {out_1std}, out_rate: {out_1std/(in_1std+out_1std)}"
            # )
            # ### [END OF ANALYSIS: in bound or not?]

            # #### [ANALYSIS: distance between poison and clean]
            # full_distance = (
            #     None
            #     if full_trigger_features_mean is None
            #     else np.linalg.norm(full_trigger_features_mean - clean_features_mean)
            # )
            # partial_distance = (
            #     None
            #     if partial_trigger_features_mean is None
            #     else np.linalg.norm(partial_trigger_features_mean - clean_features_mean)
            # )
            # no_distance = (
            #     None
            #     if no_trigger_features_mean is None
            #     else np.linalg.norm(no_trigger_features_mean - clean_features_mean)
            # )

            # print(
            #     f"full_distance: {full_distance}, partial_distance: {partial_distance}, no_distance: {no_distance}"
            # )
            # #### [END OF ANALYSIS: distance between poison and clean]

            # centered
            full_mean = np.mean(vision_features, axis=0, keepdims=True)
            # get top eigenvector
            u, s, v = np.linalg.svd(vision_features - full_mean, full_matrices=False)
            eigs = v[0:1]  # [1, 512]

            # get similarity
            corrs = np.matmul(eigs, np.transpose(vision_features))  # [1, bs*n_view]
            corrs = np.linalg.norm(
                corrs, axis=0
            )  # make positive, and flatten to [bs*n_view]
            # corrs = corrs.flatten()

            # get AUROC score
            corrs_score = corrs.reshape(num_views, -1)  # [n_views, bs]
            corrs_score = torch.mean(torch.tensor(corrs_score), dim=0)  # [bs]
            score = roc_auc_score(y_true=gt, y_score=corrs_score)
            print(f"AUG_{aug}_SEED_{seed}, score: {score*100}")
            print("===================")

            # visualization
            # corrs = np.expand_dims(corrs, axis=1)  # [bs*n_view,1]
            # corrs = np.concatenate(
            #     (corrs, np.zeros_like(corrs)), axis=1
            # )  # [bs*n_view, 2]

            tsne = TSNE(
                n_components=2,
                init="pca",
            )
            content_X = np.array(vision_features)
            transformed_x = tsne.fit_transform(content_X)

            content_Y = np.zeros(shape=(bs,))
            if len(full_trigger_global_indices) > 0:
                content_Y[np.array(full_trigger_global_indices)] = 1

            if len(partial_trigger_global_indices) > 0:
                content_Y[np.array(partial_trigger_global_indices)] = 2

            if len(no_trigger_global_indices) > 0:
                content_Y[np.array(no_trigger_global_indices)] = 3

            # MAPPING: label -> color/label/shape
            category_to_label = {
                0: "clean",
                1: "poi_FULL",
                2: "poi_PART",
                3: "poi_NO",
                # 4: "top_EIG_vec",
            }

            category_to_color = {
                0: "#0D0D04",
                1: "#F63110",
                2: "#C1876B",
                3: "lightseagreen",
                # 4: "#3B03BD",
            }

            markers = {
                0: "P",
                1: "*",
                2: "*",
                3: "*",
                # 4: "s",
            }

            fig, ax = plt.subplots()
            ax.set_title(f"AUG_{aug}_SEED_{seed}")

            for category_id, label in category_to_label.items():
                mask = content_Y == category_id

                ax.scatter(
                    transformed_x[mask, 0],
                    transformed_x[mask, 1],
                    label=label,
                    c=category_to_color[category_id],
                    marker=markers[category_id],
                    s=46,
                    # s=0.8 if category_id in [0, 1, 2, 3] else 0.3,
                )

            # legend = ax.legend(loc="lower right", shadow=True)
            # legend.get_frame()
            ax.legend()
            plt.savefig(f"AUG_{aug}_SEED_{seed}.pdf")
