import numpy as np
from config import *


def HEX_to_RGB(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))


Building = HEX_to_RGB(Building)
Land = HEX_to_RGB(Land)
Road = HEX_to_RGB(Road)
Vegetation = HEX_to_RGB(Vegetation)
Water = HEX_to_RGB(Water)
Unlabeled = HEX_to_RGB(Unlabeled)

print(f"Building color: {Building}")
print(f"Land color: {Land}")
print(f"Road color: {Road}")
print(f"Vegetation color: {Vegetation}")
print(f"Water color: {Water}")
print(f"Unlabeled color: {Unlabeled}")


def rgb_to_2D_label(label):
    seg_label = np.zeros(label.shape, dtype=np.uint8)

    # find all pixels of the label that matches the RGB arrays above (e.g. building, land, etc..)
    # if matches we replace all pixels' values to specific integer (class)
    seg_label[np.all(label == Building, axis=-1)] = 0
    seg_label[np.all(label == Land, axis=-1)] = 1
    seg_label[np.all(label == Road, axis=-1)] = 2
    seg_label[np.all(label == Vegetation, axis=-1)] = 3
    seg_label[np.all(label == Water, axis=-1)] = 4
    seg_label[np.all(label == Unlabeled, axis=-1)] = 5

    seg_label = seg_label[:, :, 0]  # no need for all channels

    return seg_label







