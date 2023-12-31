# import the necessary packages
import torch
import os
from pathlib import Path


# base path of the dataset
DATASET_PATH = "../Semantic segmentation dataset"

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 6
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
learning_rate = 0.001
EPOCHS = 50
BATCH_SIZE = 4

# define the input image dimensions
image_size = (224, 224)
PATCH_SIZE = 224

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "D:/Software/CV_Projects/Semantic_segmentation_of_aerial_imagery/Utils/output"

# define the path to the artifacts output directory
ARTIFACTS_OUTPUT = os.path.join(BASE_OUTPUT, "artifacts")
# ARTIFACTS_OUTPUT.mkdir(parents=True, exist_ok=True)

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(ARTIFACTS_OUTPUT, "unet_tgs_salt.pth")
METRIC_PLOT_PATH = os.path.join(ARTIFACTS_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(ARTIFACTS_OUTPUT, "test_paths.txt")


# these HEX are provided by kaggle dataset
"""
Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

"""

Building = '3C1098'
Land = '8429F6'
Road = '6EC1E4'
Vegetation = 'FEDD3A'
Water = 'E2A929'
Unlabeled = '9B9B9B'


""" Dataset Configs"""
train_ratio = 0.8
SEED = 1234
valid_ratio = 0.1

