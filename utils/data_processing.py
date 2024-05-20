from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def read_images(image_path_list):
    images = []
    for image_path in tqdm(image_path_list):
        image = Image.open(image_path).convert('RGB')
        images.append(image)
    return images


def process_images(images, vis_processors, device):
    processed_images = []
    for image in images:
        processed_image = vis_processors["eval"](image).unsqueeze(0).to(device)
        processed_images.append(processed_image)
    return processed_images


def read_semantics(semantics_path_list):
    semantics_list = []

    for semantic_path in tqdm(semantics_path_list):
        semantics = Image.open(semantic_path)

        # NOTE Might want to re-utilize this code on both this and the realsim-24 repos
        if semantics.mode == "RGB":
            # Selection of R channel based on https://stackoverflow.com/a/51555134
            semantics = [(d[0], 0, 0) for d in semantics.getdata()]

        # Dealing with label images not in uint8 dtype
        if semantics.mode != "L":
            semantics_np = np.array(semantics)
            semantics_np = (semantics_np >> 8).astype(np.uint8)
            semantics = torch.from_numpy(semantics_np).unsqueeze(0)
        else:
            semantics = transforms.PILToTensor()(semantics)

        semantics_list.append(semantics)

    return semantics_list


def remap_classes(semantics, map_class_nums):
    index = torch.bucketize(semantics.ravel(), map_class_nums[0])
    return map_class_nums[1][index].reshape(semantics.shape)
