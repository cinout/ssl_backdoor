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


seeds = [20, 30, 40, 42, 50]

clean_images_sim = []

poisoned_sims = []
poisoned_keys = [
    "inter_full_trigger_sims",
    "inter_partial_trigger_sims",
    "inter_no_trigger_sims",
    "cross_full_partial_sims",
    "cross_full_no_sims",
    "cross_partial_no_sims",
]

for seed in seeds:
    with open(f"VAR_seed{seed}.npy", "rb") as f:
        results = np.load(f, allow_pickle=True)  # a numpy array, not a dict
        results = results[()]  # a dict

        clean_images_sim.extend(results["inter_clean_sims"])

        poison_data = {}
        overall_sims = []
        for key_name in poisoned_keys:
            data = results[key_name]
            poison_data[key_name] = data
            overall_sims.extend(data)
        poison_data["overall_sims"] = overall_sims

        poisoned_sims.append(poison_data)

poisoned_keys.append("overall_sims")


labels = [
    "clean_1",
    "clean_2",
    "clean_3",
    "clean_4",
    "clean_5",
    "full",
    "partial",
    "no",
    "full_partial",
    "full_no",
    "partial_no",
    "overall",
]
colors = [
    "silver",
    "silver",
    "silver",
    "silver",
    "silver",
    "slateblue",
    "darkslateblue",
    "mediumpurple",
    "darkviolet",
    "plum",
    "violet",
    "royalblue",
]

for i in range(len(seeds)):

    poisoned = poisoned_sims[i]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    ax.set_ylabel("similarity")
    ax.set_title(f"5 clean images vs. Poisoned_{i+1}")

    randomly_clean_indices = random.sample(list(range(len(clean_images_sim))), k=5)

    bplot = ax.boxplot(
        [
            *[clean_images_sim[id] for id in randomly_clean_indices],
            *[poisoned[key_name] for key_name in poisoned_keys],
        ],
        patch_artist=True,
        labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    plt.savefig(f"box_plot_sd{seeds[i]}.pdf")
