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

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


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

    print(f'torch version: {torch.__version__}')
    gc.collect()
    torch.cuda.empty_cache()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment("Aerial-Imaginary-Segmentation-1")
    print(f"experiments: '{mlflow.search_experiments()}'")

    print("training started!")
    with mlflow.start_run():
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

        best_valid_loss = float('inf')

        params = {"epochs": EPOCHS, "learning_rate": learning_rate, "criterion": criterion, "optimizer": optimizer,
                  "random_state": SEED}
        mlflow.log_params(params)

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
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:6.2f}%')

        mlflow.log_metric("Train Loss", round(train_loss, 3))
        mlflow.log_metric("Train Acc", round(train_acc.item() * 100, 2))
        mlflow.log_metric("Valid Loss", round(valid_loss, 3))
        mlflow.log_metric("Valid Acc", round(valid_acc.item() * 100, 2))
        mlflow.pytorch.log_model(model, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

    print(f"experiments: '{mlflow.search_experiments()}'")

    client = MlflowClient("http://127.0.0.1:5000")

    try:
        print(f" Registered Models: {client.search_registered_models()}")
    except MlflowException:
        print("It's not possible to access the model registry :(")

    run_id = client.search_runs(experiment_ids='1')[0].info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name='UNet-Segmentation'
    )

    try:
        print(f" Registered Models: {client.search_registered_models()}")
    except MlflowException:
        print("It's not possible to access the model registry :(")
