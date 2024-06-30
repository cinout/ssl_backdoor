# import misc
import torch
import torch.nn as nn


def get_pair_wise_distance(
    data, reference, compute_mode="use_mm_for_euclid_dist_if_necessary"
):
    b = data.shape[0]
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    # offset = misc.get_rank() * b
    offset = 0
    mask = torch.zeros((b, reference.shape[0]), device=data.device, dtype=torch.bool)
    mask = torch.diagonal_scatter(mask, torch.ones(b), offset)
    r = r[~mask].view(b, -1)
    return r


# use this
class KDistanceDetector(nn.Module):
    def __init__(
        self,
        k=32,
        gather_distributed=False,
        compute_mode="use_mm_for_euclid_dist_if_necessary",
    ):
        super(KDistanceDetector, self).__init__()
        self.k = k
        self.gather_distributed = gather_distributed
        self.compute_mode = compute_mode

    def forward(self, model, images, texts=None):
        # Single modality
        vision_features = model(images)  # shape: [bs, 512]

        full_rank_vision_reference = vision_features
        d = get_pair_wise_distance(
            vision_features,
            full_rank_vision_reference,
            compute_mode=self.compute_mode,
        )

        a, _ = torch.sort(d, dim=1)
        return a[:, self.k]
