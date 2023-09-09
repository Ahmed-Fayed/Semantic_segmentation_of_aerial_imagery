import torch.nn as nn
import torch.onnx

from tqdm import tqdm
import random
import numpy as np
import time
import gc
from Utils.config import *
from Utils.utils import epoch_time

from Utils.models import UNet
from Utils.dataset import train_iterator, val_iterator, test_iterator


print(f'torch version: {torch.__version__}')
gc.collect()
torch.cuda.empty_cache()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Model Initialization
model = UNet(num_classes=NUM_CLASSES)
print(f"UNet Summery: {model}")

# loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

model = model.to(device)
criterion = criterion.to(device)


# def calculate_topk_accuracy(y_pred, y, k = 3):
#     with torch.no_grad():
#         batch_size = y.shape[0]
#         _, top_pred = y_pred.topk(k, 1)
#         top_pred = top_pred.t()
#         correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
#         correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         acc_1 = correct_1 / batch_size
#         acc_k = correct_k / batch_size
#     return acc_1, acc_k


def calculate_pixel_accuracy(preds, labels):
    """
    Calculate pixel-wise accuracy for semantic segmentation.

    Args:
    - preds (torch.Tensor): Predicted segmentation masks of shape (batch_size, num_classes, height, width).
    - labels (torch.Tensor): Ground truth segmentation masks of shape (batch_size, height, width).

    Returns:
    - accuracy (float): Pixel-wise accuracy.
    """
    # Get the predicted class labels for each pixel
    _, predicted_classes = torch.max(preds, dim=1)  # Get the class index with the highest score

    # Calculate pixel-wise accuracy
    correct_pixels = (predicted_classes == labels).float()
    accuracy = correct_pixels.sum() / labels.numel()

    return accuracy


def train(train_iterator, model, criterion, optimizer, device):
    train_size = len(train_iterator)
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in train_iterator:
        # compute predictions and loss
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        accuracy = calculate_pixel_accuracy(y_pred, y)

        # Backprobagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy

    epoch_loss /= train_size
    epoch_acc /= train_size

    return epoch_loss, epoch_acc


def evaluate(val_iterator, model, criterion, device):
    val_size = len(val_iterator)
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in val_iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred= model(x)

            loss = criterion(y_pred, y)

            accuracy= calculate_pixel_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += accuracy

    epoch_loss /= val_size
    epoch_acc /= val_size

    return epoch_loss, epoch_acc


if __name__ == "__main__":

    best_valid_loss = float('inf')

    for epoch in tqdm(range(EPOCHS)):

        start_time = time.monotonic()

        train_loss, train_acc = train(train_iterator, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(val_iterator, model, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'UNet.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc * 100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc * 100:6.2f}%')

