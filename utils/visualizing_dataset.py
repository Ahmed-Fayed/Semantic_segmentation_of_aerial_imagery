import os
import matplotlib.pyplot as plt
from config import ARTIFACTS_OUTPUT
from dataset import labels_2d
import cv2

idx = 1
# helper function for image visualization
def display(img, label, img_title, label_title, save=True):
    """
    Plot images in one row
    """
    plt.figure(figsize=(12, 6))
    plt.xticks([]);
    plt.yticks([])

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(img_title)

    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.title(label_title)

    if not os.path.exists(ARTIFACTS_OUTPUT):
        os.mkdir(ARTIFACTS_OUTPUT)

    global idx
    fig_path = os.path.join(ARTIFACTS_OUTPUT, "vis_" + str(idx) + ".jpg")
    plt.savefig(fig_path, bbox_inches='tight')
    idx += 1

    plt.show()


# Visualize example
original_image = cv2.imread("output/dataset/imgs/0005.jpg")
ground_truth_mask = cv2.imread("output/dataset/masks/0005.png")

display(original_image, ground_truth_mask, "Original Image", "Ground Truth Mask")

display(original_image, labels_2d[5], "Original Image", "2D Label")


