import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101
)
import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm

# data augmentation and normalizaation
def train_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    return torchvision_T.Compose([
        torchvision_T.ToTensor(),
        torchvision_T.RandomGrayscale(p=0.4),
        torchvision_T.Normalize(mean, std),
    ])

def common_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    return torchvision_T.Compose([
        torchvision_T.ToTensor(),
        torchvision_T.Normalize(mean, std),
    ])

# custom dataset class for training
class SegDataset(Dataset):
    def __init__(self, *, img_paths, mask_paths, image_size=(512, 512), data_type="train"):
        self.data_type = data_type
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transforms = train_transforms() if data_type == "train" else common_transforms()

    def read_file(self, path):
        file = cv2.imread(path)[:, :, ::-1]  # Convert BGR to RGB
        file = cv2.resize(file, self.image_size, interpolation=cv2.INTER_NEAREST)
        return file

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        print(f"Loading: {os.path.basename(image_path)} | {os.path.basename(mask_path)}")

        image = self.read_file(image_path)
        image = self.transforms(image)

        # convert binary mask
        gt_mask = self.read_file(mask_path).astype(np.int32)
        _mask = np.zeros((*self.image_size, 2), dtype=np.float32)
        _mask[:, :, 0] = np.where(gt_mask[:, :, 0] == 0, 1.0, 0.0)       # Background
        _mask[:, :, 1] = np.where(gt_mask[:, :, 0] == 255, 1.0, 0.0)     # Foreground
        mask = torch.from_numpy(_mask).permute(2, 0, 1)
        return image, mask

# load and prepare deeplabv3 model
def prepare_model(backbone_model="r50", num_classes=2):
    weights = 'DEFAULT'
    if backbone_model == "mbv3":
        model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
    elif backbone_model == "r50":
        model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    elif backbone_model == "r101":
        model = deeplabv3_resnet101(weights=weights, aux_loss=True)
    else:
        raise ValueError("Backbone must be one of 'mbv3', 'r50', or 'r101'")

    # Replace final classifiers with LazyConv for 2-class output
    model.classifier[4] = nn.LazyConv2d(num_classes, 1)
    model.aux_classifier[4] = nn.LazyConv2d(num_classes, 1)
    return model

# metric calculations
def intermediate_metric_calculation(predictions, targets, use_dice=False, smooth=1e-6, dims=(2,3)):
    intersection = (predictions * targets).sum(dim=dims) + smooth
    summation = predictions.sum(dim=dims) + targets.sum(dim=dims) + smooth
    if use_dice:
        metric = (2.0 * intersection) / summation
    else:
        union = summation - intersection
        metric = intersection / union
    return metric.mean()

def convert_2_onehot(predictions, num_classes):
    onehot = torch.zeros_like(predictions)
    max_vals, indices = predictions.max(dim=1, keepdim=True)
    onehot.scatter_(1, indices, 1.0)
    return onehot

def total_variation_loss(pred):
    diff_i = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    diff_j = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    return (diff_i.mean() + diff_j.mean())

# composite loss function
class SegLoss:
    def __init__(self, use_dice=True, smooth=1e-6, num_classes=2, tv_weight=0.1):
        self.use_dice = use_dice
        self.smooth = smooth
        self.num_classes = num_classes
        self.tv_weight = tv_weight

    def __call__(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        pixel_loss = F.binary_cross_entropy(predictions, targets, reduction="mean")
        mask_loss = 1 - intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)
        tv_loss = total_variation_loss(predictions)
        total_loss = pixel_loss + mask_loss + self.tv_weight * tv_loss

        onehot_preds = convert_2_onehot(predictions, num_classes=self.num_classes)
        metric = intermediate_metric_calculation(onehot_preds, targets, use_dice=self.use_dice, smooth=self.smooth)
        return total_loss, metric

# === Training Script ===
if __name__ == "__main__":
    # load dataset
    image_folder = "synthetic_dataset/images"
    mask_folder = "synthetic_dataset/masks"
    img_list = sorted(glob(os.path.join(image_folder, "*.jpg")))
    mask_list = sorted(glob(os.path.join(mask_folder, "*.png")))

    assert len(img_list) == len(mask_list), "Mismatch in number of images and masks!"
    for i in range(3):  # Show a sample of file pairs
        print(img_list[i], "<->", mask_list[i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init dataset and dataloadder instances
    dataset = SegDataset(img_paths=img_list, mask_paths=mask_list)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # model, loss and scheduler init
    model = prepare_model("r50", num_classes=2).to(device)
    criterion = SegLoss(tv_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_metric = 0.0

        # training loop
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss, metric = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metric += metric.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_metric = total_metric / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}, Metric: {avg_metric:.4f}")

    # Save trained model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/document_segmenter.pth")
    print("Model saved to 'checkpoints/document_segmenter.pth'")
