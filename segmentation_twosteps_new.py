# In this file, we define the CNN model for segmentation tasks.

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision import transforms

from torch.utils.data import DataLoader, random_split
from GeneralDataset import TectonicPatchDataset_multiclass
from glob import glob
from segmentation_models_pytorch.utils.metrics import IoU
from tqdm import tqdm
import os
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime

from unetwithpe import UNetWithPE, UNetWithPEAndAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
today_str = datetime.today().strftime("%y%m%d")
print(today_str)
trace_prob = 7e-1
encoder_name = "resnet34"
binary_model_path = "logs/general-borehole-cnn/Unet-resnet34-binary-faciesbalance-dcn-ep200-lr0.0001-260716-trace0.7.pth"

wandb.init(
    project="general-boreholes-segmentation-multiclass-2step-classbalance",
    config={
        "model": f"Unet-{encoder_name}-multiclass-faciesbalance-dcn-celoss-oldBN",
        "lr": 1e-3,
        "batch_size": 24,
        "val_batch_size": 4,
        "epochs": 250,
        "loss": "CEloss",
        "optimizer": "Adam",
        "trace_prob": f"{trace_prob}",
    }
)
num_class = 8

def get_image_and_mask_paths(optv_dir, label_dir):
    # Load all files from both folders
    image_files = glob(os.path.join(optv_dir, "*"))
    label_files = glob(os.path.join(label_dir, "*"))

    # Build lookup by stem (e.g., '00123')
    label_map = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

    image_paths = []
    mask_paths = []

    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem in label_map:
            image_paths.append(img_path)
            mask_paths.append(label_map[stem])
        else:
            print(f"No label found for {stem}")

    print(f"Found {len(image_paths)} matched image-mask pairs")
    return image_paths, mask_paths

optv_dir = "data/general-boreholes-training/optv_faciesbalance"
label_dir = "data/general-boreholes-training/label_multiclass_faciesbalance"
# label_dir = "data/general-boreholes-training/label_lowershaly"

image_paths, mask_paths = get_image_and_mask_paths(optv_dir, label_dir)

transform = A.Compose([
    A.Resize(256, 256),  # Resize to fixed size
    # A.HorizontalFlip(p=0.3),  # Random horizontal flip
    # A.VerticalFlip(p=0.3),  # Random vertical flip
    # A.RandomRotate90(p=0.3),  # Random rotation
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for ResNet
    ToTensorV2(),  # Convert to PyTorch tensor
])

dataset = TectonicPatchDataset_multiclass(
    image_paths=image_paths,
    mask_paths=mask_paths,
    patch_sizes=[256,384,512,640,768,1080,1440],  # Different patch sizes
    samples_per_image=200,
    trace_prob = trace_prob,
    transform=transform,
    add_coords=True,
    # device=device
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Step 3: Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Example model
# model = Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
iou_metric = IoU(threshold=0.5)

best_val_iou = 0
save_path = f"logs/faciesbalance/twostep-{wandb.config.model}-ep{wandb.config.epochs}-lr{wandb.config.lr}-{today_str}-trace{wandb.config.trace_prob}_trainboth_{encoder_name}.pth"


import torch
import torch.nn.functional as F

def loss_with_inversion_multiclass(
    y_true,
    y_pred,
    eps=1e-6,
    alpha=0.5,
    class_weights=None,
    background_idx=0,
    invert_to_class=1,
):
    """
    Combined CrossEntropy + Dice loss with background-patch inversion for multi-class segmentation.

    Args:
        y_true: [B, H, W] tensor with ground truth labels in [0..C-1]
        y_pred: [B, C, H, W] tensor with raw logits
        eps: small value to avoid division by zero
        alpha: weight for CE (1 - alpha for Dice)
        class_weights: optional tensor [C] with weights for each class (for CE)
        background_idx: which class index is background
        invert_to_class: class index used as "fake foreground" when patch is pure background
                        (must be != background_idx). Default assumes class 1 exists.
    Returns:
        scalar loss
    """
    B, C, H, W = y_pred.shape
    if C < 2:
        raise ValueError("Inversion requires at least 2 classes (C >= 2).")
    if not (0 <= background_idx < C):
        raise ValueError(f"background_idx must be in [0, {C-1}]")
    if not (0 <= invert_to_class < C):
        raise ValueError(f"invert_to_class must be in [0, {C-1}]")
    if invert_to_class == background_idx:
        raise ValueError("invert_to_class must be different from background_idx.")

    losses = []

    for i in range(B):
        yi = y_true[i].long()     # [H, W]
        yhi = y_pred[i]           # [C, H, W], logits

        # --- Check if patch is pure background ---
        if torch.all(yi == background_idx):
            # Fake target: pick a non-background class everywhere
            yi_inv = torch.full_like(yi, fill_value=invert_to_class)

            # Build "inverted" logits:
            # - Make background logit very unfavorable (using -bg_logit)
            # - Set all non-background logits to 0 (uniform among foreground)
            bg_logit = yhi[background_idx]                    # [H, W]
            inv_logits = torch.zeros_like(yhi)                # [C, H, W]
            inv_logits[background_idx] = -bg_logit            # discourage background

            ce = F.cross_entropy(
                inv_logits.unsqueeze(0),
                yi_inv.unsqueeze(0),
                weight=class_weights,
            )

            probs = torch.softmax(inv_logits, dim=0)          # [C, H, W]
            probs_flat = probs.reshape(C, -1)
            yi_onehot = F.one_hot(yi_inv, num_classes=C).permute(2, 0, 1).reshape(C, -1).float()

        else:
            # Normal patch: standard CE + multi-class Dice
            ce = F.cross_entropy(
                yhi.unsqueeze(0),
                yi.unsqueeze(0),
                weight=class_weights,
            )

            probs = torch.softmax(yhi, dim=0)                 # [C, H, W]
            probs_flat = probs.reshape(C, -1)
            yi_onehot = F.one_hot(yi, num_classes=C).permute(2, 0, 1).reshape(C, -1).float()

        intersection = (probs_flat * yi_onehot).sum(dim=1)
        union = probs_flat.sum(dim=1) + yi_onehot.sum(dim=1)
        dice_per_class = 1 - (2 * intersection + eps) / (union + eps)
        dice = dice_per_class.mean()

        loss = alpha * ce + (1 - alpha) * dice
        losses.append(loss)

    return torch.stack(losses).mean()

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # [B,C,H,W]

        valid_mask = (targets != self.ignore_index)
        targets = targets.clone()
        targets[~valid_mask] = 0

        one_hot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        present = one_hot.sum(dims) > 0                    # which classes exist in this batch
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        dice_present = dice[present]
        return 1.0 - (dice_present.mean() if present.any() else dice.mean())        
        # return 1.0 - dice.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        valid = (targets != self.ignore_index)
        return focal[valid].mean()

def unnormalize(image_tensor, mean, std):
    """
    Accepts a torch.Tensor [C, H, W] or a NumPy array [C, H, W].
    Returns a NumPy array [C, H, W] with values unnormalized.
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.clone().detach().cpu().numpy()
    elif isinstance(image_tensor, np.ndarray):
        image = image_tensor.copy()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray but got {type(image_tensor)}")

    # Unnormalize each channel
    for c in range(image.shape[0]):
        image[c] = image[c] * std[c] + mean[c]
    return image


def visualize_batch(images, masks, preds=None, max_samples=4):
    """
    Visualize a batch of images and masks (and optionally predictions).
    Assumes inputs are tensors: [B, C, H, W] and [B, H, W]
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    if preds is not None:
        preds = torch.sigmoid(preds).detach().cpu().numpy() > 0.5

    batch_size = min(len(images), max_samples)
    fig, axes = plt.subplots(batch_size, 3 if preds is not None else 2, figsize=(12, 4 * batch_size))

    for i in range(batch_size):
        img = images[i].transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
        # Unnormalize the image
        # mean=(0.485, 0.456, 0.406),
        # std=(0.229, 0.224, 0.225)

        # img = unnormalize(images[i], mean, std).transpose(1, 2, 0)  # [H, W, C]
        # img = np.clip(img, 0, 1) 
        gt = masks[i]
        if img.shape[2] == 1:  # grayscale
            img = img[:, :, 0]

        if preds is not None:
            pred = preds[i][0] if preds[i].ndim == 3 else preds[i]
        
        if batch_size == 1:
            axes_row = axes
        else:
            axes_row = axes[i]

        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes_row[0].imshow(img_norm, cmap='gray' if img.ndim == 2 else None)
        axes_row[0].set_title("Image")

        axes_row[1].imshow(gt, cmap='gray')
        axes_row[1].set_title("Ground Truth")

        if preds is not None:
            axes_row[2].imshow(pred, cmap='gray')
            axes_row[2].set_title("Prediction")

        for ax in axes_row:
            ax.axis("off")

    plt.tight_layout()
    return fig

class FrozenBinaryGuidedMulti_old(nn.Module):
    """
    Stage-2 refiner guided by the Stage-1 binary model.
    Binary probability map is concatenated as an extra input channel.

    finetune_binary=True  → binary weights receive gradients (use small LR group),
                            but its BatchNorm statistics stay FROZEN so the
                            pretrained normalisation is preserved.
    finetune_binary=False → binary fully frozen, forward under no_grad
                            (faster, less memory).
    """
    def __init__(self, binary_model: nn.Module, num_classes: int, in_channels: int,
                 encoder_name="resnet34", encoder_weights="imagenet",
                 finetune_binary=True):
        super().__init__()
        self.binary = binary_model
        self.finetune_binary = finetune_binary
        for p in self.binary.parameters():
            p.requires_grad = finetune_binary

        self.refiner = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels + 1,      # +1 for binary probability map
            classes=num_classes,
        )

    def train(self, mode=True):
        """Keep binary's BN layers in eval mode even during training,
        so running statistics from Stage 1 are never overwritten."""
        super().train(mode)
        if mode:
            for m in self.binary.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        return self

    def forward(self, x):
        # x: [B,6,H,W] = RGB + PE
        if self.finetune_binary:
            bin_logits = self.binary(x)                   # grads flow (small LR)
        else:
            with torch.no_grad():
                bin_logits = self.binary(x)

        bin_prob = torch.sigmoid(bin_logits)              # bounded [0,1] guidance
        x_guided = torch.cat([x, bin_prob], dim=1)        # [B,7,H,W]
        return self.refiner(x_guided)                     # [B,num_classes,H,W]

class FrozenBinaryGuidedMulti(nn.Module):
    def __init__(self, binary_model: nn.Module, num_classes: int, in_channels: int,
                 encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        self.binary = binary_model
        # self.binary.eval()
        for p in self.binary.parameters():
            p.requires_grad = True

        # Trainable refiner (can be SMP Unet)
        self.refiner = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels + 1,  # +1 channel for binary logits ; +16 for feature maps
            classes=num_classes
        )

    def forward(self, x):
        # x: [B,6,H,W] = RGB+PE
        # with torch.no_grad():
        #     bin_logits, bin_feat = self.binary(x)  # expected [B,1,H,W] logits
            # if your binary model returns [B,H,W], fix it to return [B,1,H,W]
        bin_logits = self.binary(x) 

        # x_rgb = x[:,:3,:,:]  # [B,3,H,W]
        x_guided = torch.cat([x, bin_logits], dim=1)  # [B,7,H,W]
        ###### use the feature maps from binary model ########        
        # x_guided = torch.cat([x, bin_feat], dim=1)  # [B,7,H,W]
        mc_logits = self.refiner(x_guided)            # [B,9,H,W]
        return mc_logits

binary = UNetWithPE(encoder_name=encoder_name, encoder_weights="imagenet",
                    in_channels=6, classes=1, use_dcn=True, return_features=False).to(device)
binary.load_state_dict(torch.load(binary_model_path, map_location=device))


# --- build guided multiclass model ---
num_class = 8
model = FrozenBinaryGuidedMulti(
    binary_model=binary, 
    num_classes=num_class, 
    in_channels=6,
    # pe_channels=3,
    encoder_name=encoder_name,
    encoder_weights="imagenet",
    # finetune_binary=True,
    ).to(device)

# optimizer ONLY for refiner (binary frozen)
# optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = torch.optim.AdamW(
    [
        {"params": model.refiner.parameters(), "lr": 1e-3},
        {"params": [p for p in model.binary.parameters() if p.requires_grad], "lr": 1e-5},
    ],
    weight_decay=1e-2
)



def mean_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    logits: [B, C, H, W]
    target: [B, H, W] with values in [0..C-1]
    """
    pred = logits.argmax(dim=1)  # [B,H,W]
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        tgt_c = (target == c)
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union == 0:
            # if class absent in both pred and gt, skip it (common choice)
            continue
        ious.append((inter + eps) / (union + eps))
    if len(ious) == 0:
        return torch.tensor(1.0, device=logits.device)  # nothing present -> perfect by convention
    return torch.stack(ious).mean()

def prep_masks(masks):
    """Shared by train AND val — fixes the label-space mismatch."""
    masks = masks.clone()
    masks[masks == 8] = 0
    return masks.clamp(0, num_class - 1).long()

best_val_iou, stale, PATIENCE = 0.0, 0, 25
class_weights = torch.tensor([1.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0], device=device)
loss_ce = nn.CrossEntropyLoss()
loss_dice = MultiClassDiceLoss(num_classes=num_class)
loss_focal = FocalLoss(gamma=2.0)
scaler  = torch.amp.GradScaler("cuda")

for epoch in range(wandb.config.epochs):
    # ---------------- TRAIN ----------------
    model.train()          # binary BN stays frozen via the train() override
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"epoch {epoch}"):
        images = images.to(device, non_blocking=True)
        masks  = prep_masks(masks).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss  = loss_ce(preds, masks)
            # loss = 0.5 * loss_focal(preds, masks) + 0.5 * loss_dice(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---------------- VAL ----------------
    model.eval()
    val_iou = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = prep_masks(masks).to(device, non_blocking=True)   # ← Bug 1 fixed
            preds  = model(images)
            val_iou += mean_iou_from_logits(preds.float(), masks, num_class).item()

    avg_val_iou = val_iou / len(val_loader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_iou": avg_val_iou})
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_val_iou:
        best_val_iou, stale = avg_val_iou, 0
        torch.save(model.state_dict(), save_path)
        print(f"  Saved new best (mIoU {best_val_iou:.4f})")
    # else:
    #     stale += 1
    #     if stale >= PATIENCE:
    #         print(f"Early stopping at epoch {epoch} (best {best_val_iou:.4f})")
    #         break

'''
for epoch in range(wandb.config.epochs):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0

    for i, (images, masks) in enumerate(tqdm(train_loader)):
        images = images.to(device, non_blocking=True)               # [B,6,H,W]
        masks = prep_masks(masks)
        masks = masks.to(device, non_blocking=True).long()          # [B,H,W] class ids        

        # (optional safety) ensure valid range
        masks = masks.clamp(0, num_class - 1)

        preds = model(images)                                       # [B,C,H,W] logits

        # loss = loss_with_inversion_multiclass(
        #     y_true=masks,
        #     y_pred=preds,
        #     class_weights=class_weights,
        #     background_idx=0,
        #     invert_to_class=1,
        # )
        # loss = 0.5 * loss_focal(preds, masks) + 0.5 * loss_dice(preds, masks)
        loss = loss_ce(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---------------- VAL ----------------
    model.eval()
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()
            masks = masks.clamp(0, num_class - 1)

            preds = model(images)                                  # [B,C,H,W]

            val_iou += mean_iou_from_logits(preds, masks, num_class).item()

    avg_val_iou = val_iou / len(val_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_iou": avg_val_iou
    })

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model at epoch {epoch} with mIoU {best_val_iou:.4f}")
'''
