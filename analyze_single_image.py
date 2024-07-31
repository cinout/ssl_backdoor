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

num_views = 64  # FIXME: update
gt = np.array([1, 0, 0, 0, 0, 0, 0, 0])

for aug in augs:
    for seed in seeds:
        with open(f"SS_AUG_{aug}_SEED_{seed}_STEP_1.npy", "rb") as f:
            results = np.load(f, allow_pickle=True)  # a numpy array, not a dict
            results = results[()]  # a dict

            full_trigger_global_indices = results["full_trigger_global_indices"]
            partial_trigger_global_indices = results["partial_trigger_global_indices"]
            no_trigger_global_indices = results["no_trigger_global_indices"]

            vision_features = results["vision_features"]
            total_views, channel = vision_features.shape

            poison_features = None
            poison_labels = None
            if len(full_trigger_global_indices) > 0:
                full_trigger_features = vision_features[
                    np.array(full_trigger_global_indices)
                ]
                if poison_features is None:
                    poison_features = full_trigger_features
                    poison_labels = [0] * full_trigger_features.shape[0]

            if len(partial_trigger_global_indices) > 0:
                partial_trigger_features = vision_features[
                    np.array(partial_trigger_global_indices)
                ]
                if poison_features is None:
                    poison_features = partial_trigger_features
                    poison_labels = [1] * partial_trigger_features.shape[0]
                else:
                    poison_features = np.concatenate(
                        [poison_features, partial_trigger_features], axis=0
                    )
                    poison_labels = (
                        poison_labels + [1] * partial_trigger_features.shape[0]
                    )

            if len(no_trigger_global_indices) > 0:
                no_trigger_features = vision_features[
                    np.array(no_trigger_global_indices)
                ]
                if poison_features is None:
                    poison_features = no_trigger_features
                    poison_labels = [2] * no_trigger_features.shape[0]
                else:
                    poison_features = np.concatenate(
                        [poison_features, no_trigger_features], axis=0
                    )
                    poison_labels = poison_labels + [2] * no_trigger_features.shape[0]

            tsne = TSNE(
                n_components=2,
                init="pca",
            )
            transformed_x = tsne.fit_transform(poison_features)
            content_Y = np.array(poison_labels)

            # MAPPING: label -> color/label/shape
            category_to_label = {
                0: "poi_FULL",
                1: "poi_PART",
                2: "poi_NO",
            }

            category_to_color = {
                0: "#F63110",
                1: "#C1876B",
                2: "lightseagreen",
            }

            markers = {
                0: "P",
                1: "*",
                2: "*",
            }

            fig, ax = plt.subplots()
            ax.set_title(f"AUG_{aug}_SEED_{seed}_POISON")

            for category_id, label in category_to_label.items():
                mask = content_Y == category_id

                ax.scatter(
                    transformed_x[mask, 0],
                    transformed_x[mask, 1],
                    label=label,
                    c=category_to_color[category_id],
                    marker=markers[category_id],
                    s=72,
                    # s=0.8 if category_id in [0, 1, 2, 3] else 0.3,
                )

            # legend = ax.legend(loc="lower right", shadow=True)
            # legend.get_frame()
            ax.legend()
            plt.savefig(f"AUG_{aug}_SEED_{seed}_poison.pdf")

            bs = gt.shape[0]
            num_views = int(total_views / bs)
            poison_index = gt.tolist().index(1)
            for i in range(bs):
                if i != poison_index:
                    clean_views_indices = [
                        view_i * bs + i for view_i in range(num_views)
                    ]

                    clean_features = vision_features[np.array(clean_views_indices)]
                    transformed_x = tsne.fit_transform(clean_features)
                    content_Y = np.zeros(clean_features.shape[0])

                    # MAPPING: label -> color/label/shape
                    category_to_label = {
                        0: "clean",
                    }

                    category_to_color = {
                        0: "lightseagreen",
                    }

                    markers = {
                        0: "*",
                    }

                    fig, ax = plt.subplots()
                    ax.set_title(f"AUG_{aug}_SEED_{seed}_CLEAN_{i}")

                    for category_id, label in category_to_label.items():
                        mask = content_Y == category_id

                        ax.scatter(
                            transformed_x[mask, 0],
                            transformed_x[mask, 1],
                            label=label,
                            c=category_to_color[category_id],
                            marker=markers[category_id],
                            s=72,
                            # s=0.8 if category_id in [0, 1, 2, 3] else 0.3,
                        )

                    ax.legend()
                    plt.savefig(f"AUG_{aug}_SEED_{seed}_CLEAN_{i}.pdf")
