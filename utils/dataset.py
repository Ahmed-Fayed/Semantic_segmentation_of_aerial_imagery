import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from patchify import patchify

from Utils.config import *
from Utils.utils import rgb_to_2D_label

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


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

# labels_2d = np.array(labels_2d)
# labels_2d = np.expand_dims(labels_2d, axis=3)

# print("Unique labels in label dataset are: ", np.unique(labels_2d))


""" Create train and test dataset"""

print("loading dataset..")
dataset_imgs_dir = "D:/Software/CV_Projects/Semantic_segmentation_of_aerial_imagery/utils/output/dataset/imgs"
dataset_images = []
for img_name in os.listdir(dataset_imgs_dir):
    img_path = os.path.join(dataset_imgs_dir, img_name)
    dataset_images.append(img_path)

train_data, test_data, train_labels, test_labels = train_test_split(dataset_images, labels_2d, test_size=0.1,
                                                                    shuffle=True, random_state=SEED)

train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.1,
                                                                      shuffle=True, random_state=SEED)

print(f"train_data: {train_data[5]}")
print(f"train_labels: {train_labels[5]}")
print(f"train_data: {valid_data[5]}")
print(f"train_labels: {valid_labels[5]}")
print(f"test_data: {test_data[5]}")
print(f"test_labels: {test_labels[5]}")


# Define a custom dataset class
# class CustomImageDataset(Dataset):
#     def __init__(self, image_paths, label_images=None, transform=None):
#         self.image_paths = image_paths
#         self.label_images = label_images
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         # image = cv2.imread(img_path)
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         if self.transform:
#             image = self.transform(image)
#
#         if self.label_images is not None:
#             label_image = self.label_images[idx]
#             # label_image = Image.fromarray(label_image, mode="L")  # Convert NumPy array to PIL Image
#
#             if self.transform:
#                 label_image = self.transform(label_image)
#
#             return image, label_image
#         else:
#             return image


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_list, transform=None):
        self.image_paths= image_paths
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        # Load the image
        img_filename = self.image_paths[idx]
        image = Image.open(img_filename).convert("RGB")

        # Load the mask from the list of NumPy arrays
        mask = self.mask_list[idx]

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()  # Convert the mask to a PyTorch tensor

        return image, mask


train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(image_size, padding=10),
    transforms.ToTensor(),
    # transforms.Normalize(mean=target_means, std=target_stds),
    transforms.RandomErasing(p=0.2)

])

test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=target_means, std=target_stds)
])

# Create a custom dataset instance
train_dataset = CustomSegmentationDataset(train_data, train_labels, train_transforms)
valid_dataset = CustomSegmentationDataset(valid_data, valid_labels, train_transforms)
test_dataset = CustomSegmentationDataset(test_data, test_labels, test_transforms)

# Create a data loader
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Iterate through the data loader
print(type(train_iterator))
print(train_iterator)
images, labels = next(iter(train_iterator))
print("Batch Size:", images.size(0))
print(f"image: {images[0]}, Image Shape: {images[0].size()}")
print(f"label: {labels[0]}, Label Shape: {labels[0].size()}")

print(f'num of training examples: {len(train_data)}')
print(f'num of validation examples: {len(valid_data)}')
print(f'num of testing examples: {len(test_data)}')
