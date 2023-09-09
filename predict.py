import torch
from PIL import Image
from Utils.models import UNet
from torchvision import transforms
import torch.nn.functional as F
from Utils.config import *
from Utils.visualizing_dataset import display

model = UNet(num_classes=6)
model.load_state_dict(torch.load("UNet.pt"))
print(f"model summery: {model}")

model.eval()

test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=target_means, std=target_stds)
])

img = Image.open("D:/Software/CV_Projects/Semantic_segmentation_of_aerial_imagery/Utils/output/dataset/imgs/0005.jpg")
img = img.convert("RGB")
trans_img = test_transforms(img)
trans_img = torch.unsqueeze(trans_img, 0)

with torch.no_grad():
    y_pred = model(trans_img)
    y_prob = F.softmax(y_pred, dim=-1)
    top_pred = y_prob.argmax(1, keepdim=True)
    top_pred = torch.squeeze(top_pred)
    print(f"pred: {top_pred}")


# Visualize example
display(img, top_pred, "Original Image", "Predicted Label", "Final_Prediction")

