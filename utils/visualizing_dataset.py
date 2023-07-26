import os
import matplotlib.pyplot as plt
from config import ARTIFACTS_OUTPUT
import cv2


# helper function for image visualization
def display(**images):
    """
    Plot images in one row
    """
    # num_images = len(images)
    plt.figure(figsize=(12, 12))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, 2, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=15)
        plt.imshow(image)
        fig_path = os.path.join(ARTIFACTS_OUTPUT, "dataset_sample.jpg")
        plt.savefig(fig_path, bbox_inches='tight')

    print(f"figure saved to path: {fig_path}")
    plt.show()


# Visualize example
original_image = cv2.imread("output/dataset/imgs/0000.jpg")
ground_truth_mask = cv2.imread("output/dataset/masks/0000.png")

if not os.path.exists(ARTIFACTS_OUTPUT):
    os.mkdir(ARTIFACTS_OUTPUT)

display(original_image=original_image, ground_truth_mask=ground_truth_mask)



