import numpy as np
from tqdm import tqdm
import os
import cv2
from PIL import Image
from pathlib import Path
from config import *
from patchify import patchify
from utils import rgb_to_2D_label


def create_patches(img_path, rgb=True):
    img = cv2.imread(img_path)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get width to the nearest size divisible by patch size
    size_x = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
    # get height to the nearest size divisible by patch size
    size_y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE

    img = Image.fromarray(img)
    # Crop original image to size divisible by patch size from top left corner
    img = img.crop((0, 0, size_x, size_y))
    img = np.array(img)

    # extract patches from each image
    # step = PATCH_SIZE for patches with size PATCH_SIZE means no overlap
    img_pathes = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)

    return img_pathes


def write_patches(img_patches, patch_index, target_dir, patch_extention = ".jpg"):
    # iterate over vertical patch axis
    for j in range(img_patches.shape[0]):
        # iterate over horizontal patch axis
        for k in range(img_patches.shape[1]):
            # patches are located like a grid. use (j, k) indices to extract single patched image
            single_patch_img = img_patches[j, k, :, :]

            # Drop extra dimension from patchify
            single_patch_img = np.squeeze(single_patch_img)

            patch_file_name = os.path.join(target_dir, str(patch_index).zfill(4) + patch_extention)
            cv2.imwrite(patch_file_name, single_patch_img)
            patch_index += 1


def create_dataset(dataset_dir, target_dir):
    """

    :param dataset_dir: base dir for dataset
    :param target_dir: output dir in which we will write the dataset

    dataset_dir Structure:
    - Semantic segmentation dataset
        - Tile 1
            - images
            - masks
        - Tile 2
        ........


    """
    target_dir_path = Path(target_dir)
    target_imgs_path = Path(target_dir + "/dataset/imgs/")
    target_masks_path = Path(target_dir + "/dataset/masks/")

    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_imgs_path.mkdir(parents=True, exist_ok=True)
    target_masks_path.mkdir(parents=True, exist_ok=True)

    imgs_index, masks_index = 0, 0

    for tile in tqdm(os.listdir(dataset_dir)):

        if not tile.endswith('.json'):

            tile_path = os.path.join(dataset_dir, tile)
            tile_images_path = os.path.join(tile_path, 'images')
            tile_masks_path = os.path.join(tile_path, 'masks')

            for img_name in os.listdir(tile_images_path):

                if img_name.endswith(".jpg"):
                    img_path = os.path.join(tile_images_path, img_name)
                    img_patches = create_patches(img_path)

                    write_patches(img_patches, imgs_index, target_imgs_path, ".jpg")

            for mask_name in os.listdir(tile_masks_path):

                if mask_name.endswith(".png"):
                    mask_path = os.path.join(tile_masks_path, mask_name)
                    mask_patches = create_patches(mask_path)

                    write_patches(mask_patches, masks_index, target_masks_path, ".png")

    print(f"Dataset saved to path: {target_dir_path}")


# create_dataset(dataset_dir=DATASET_PATH, target_dir=BASE_OUTPUT)


labels_2d = []


def create_2d_labels(masks_path, target_output=""):
    for mask_name in tqdm(os.listdir(masks_path)):
        mask_img = cv2.imread(os.path.join(masks_path, mask_name))
        label = rgb_to_2D_label(mask_img)
        labels_2d.append(label)
        # print(f"label_2d: {label_2d}, type: {type(label_2d)}")
        # filepath = os.path.join(target_output, mask_name)
        # os.makedirs(target_output, exist_ok=True)
        # cv2.imwrite(filepath, label_2d)


data_dir = "D:/Software/CV_Projects/Semantic_segmentation_of_aerial_imagery/utils/output/dataset"
masks_path = os.path.join(data_dir, "masks")
# target_masks_path = os.path.join(data_dir, "labels_2d")
create_2d_labels(masks_path)

labels_2d = np.array(labels_2d)
labels_2d = np.expand_dims(labels_2d, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels_2d))
