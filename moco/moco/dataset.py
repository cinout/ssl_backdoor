import os
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms


class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        with open(path_to_txt_file, "r") as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform
        # self.debug_print_views = debug_print_views

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert("RGB")
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:

            # if self.debug_print_views:
            #     # return both PIL and final tensor
            #     raw_pils = self.transform(img)
            #     images = [to_tensor(raw) for raw in raw_pils]
            #     return image_path, (raw_pils, images), target, idx
            # else:
            images = self.transform(img)
            return image_path, images, target, idx

    def __len__(self):
        return len(self.file_list)
