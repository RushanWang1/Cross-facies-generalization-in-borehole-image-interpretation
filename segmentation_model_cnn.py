# In this file, we define the CNN model for segmentation tasks.

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision import transforms

from torch.utils.data import DataLoader, random_split, ConcatDataset
from GeneralDataset import TectonicPatchDataset_multiclass, HardNegNPZDatasetWithPE
from glob import glob
from segmentation_models_pytorch.utils.metrics import IoU
from tqdm import tqdm
import os
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from unetwithpe import UNetWithPE, UNetWithPEAndAttention, UNetWithPE_BH 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

today_str = datetime.today().strftime("%y%m%d")
print(today_str)
trace_prob = 7e-1
encoder_name = "resnet34"
wandb.init(
    project="general-boreholes-segmentation",
    config={
        "model": f"Unet-{encoder_name}-binary-sandy-dcn",
        "lr": 1e-4,
        "batch_size": 24,
        "epochs": 200,
        "loss": "BCEWithLogitsLoss-Dice-inverse",
        "optimizer": "Adam",
        "trace_prob": f"{trace_prob}",
    }
)
num_class = 1

# model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=num_class)
# Encoder name: "resnet34", "mit_b2", "efficientnet-b4"
model = UNetWithPE(encoder_name=encoder_name, encoder_weights="imagenet",in_channels=6, classes=num_class, use_dcn=True)

# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.001-260128-trace0.7_multiscale_inversionloss_dcn.pth"
# model.load_state_dict(torch.load(model_path))
# model.cuda()  # Move model to GPU
model = model.to(device)
print("Pretrained model loaded successfully.")

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
label_dir = "data/general-boreholes-training/label_sandy"
soft_dir = "data/general-boreholes-training/label_soft_binary"

image_paths, mask_paths = get_image_and_mask_paths(optv_dir, label_dir)


transform = A.Compose([
    A.Resize(256, 256),  # Resize to fixed size
    # A.HorizontalFlip(p=0.3),  # Random horizontal flip
    # A.VerticalFlip(p=0.3),  # Random vertical flip
    # A.RandomRotate90(p=0.3),  # Random rotation
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for ResNet
    ToTensorV2(),  # Convert to PyTorch tensor
], additional_targets={"soft": "mask"})

dataset = TectonicPatchDataset_multiclass(
    image_paths=image_paths,
    mask_paths=mask_paths,
    # soft_paths=soft_paths,
    patch_sizes=[256,384,512,640,768,1080,1440],  # Different patch sizes
    samples_per_image=150,
    trace_prob = trace_prob,
    transform=transform,
    add_coords=True,
    add_soft_channel=False,
    preload=True,
    background_idx=0,
    # device=device
)

# hardneg_ds = HardNegNPZDatasetWithPE(
#     npz_dir="data/general-boreholes-training/hardneg_fault_512", 
#     transform=transform,
#     # device=device
# )

# mixed_train = ConcatDataset([dataset, hardneg_ds])

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Step 3: Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# Example model
# model = Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
# loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
iou_metric = IoU(threshold=0.5)

best_val_iou = 0
save_path = f"logs/general-borehole-cnn/{wandb.config.model}-ep{wandb.config.epochs}-lr{wandb.config.lr}-{today_str}-trace{wandb.config.trace_prob}.pth"

def loss_with_inversion(y_true, y_pred, eps=1e-6, alpha=0.5, fp_weight=10.0):
    """
    Vectorised BCE + Dice with background-patch inversion.
    Identical behaviour to the original looped version, but:
      - no Python loop over the batch (single batched computation)
      - uses BCE-with-logits (numerically stable, safe under torch.amp.autocast)
      - no dead double-BCE computation

    y_true: [B, H, W] binary ground truth
    y_pred: [B, H, W] raw logits
    """
    y      = y_true.float()
    logits = y_pred.float()                      # float32 even under autocast
    B      = y.shape[0]

    is_bg = (y.flatten(1).sum(dim=1) == 0)       # [B] background-only patches

    # ---- BCE ----
    # Normal patches: weighted BCE. Background patches: BCE(1-p, 1) == BCE(p, 0),
    # i.e. the same target y (all zeros), just unweighted — so one batched call.
    weights = torch.where(y == 0,
                          torch.full_like(y, fp_weight),
                          torch.ones_like(y))
    weights[is_bg] = 1.0                          # bg patches unweighted (as original)

    bce = F.binary_cross_entropy_with_logits(logits, y, weight=weights,
                                             reduction='none')
    bce = bce.flatten(1).mean(dim=1)              # [B] per-sample mean

    # ---- Dice ----
    # Normal: dice(p, y). Background: dice(1-p, 1-y).
    p     = torch.sigmoid(logits)
    flip  = is_bg.view(B, 1, 1)
    p_eff = torch.where(flip, 1.0 - p, p)
    y_eff = torch.where(flip, 1.0 - y, y)

    inter = (p_eff * y_eff).flatten(1).sum(dim=1)
    denom = p_eff.flatten(1).sum(dim=1) + y_eff.flatten(1).sum(dim=1)
    dice  = 1.0 - (2.0 * inter + eps) / (denom + eps)   # [B]

    return (alpha * bce + (1.0 - alpha) * dice).mean()

def loss_with_inversion_multiclass(
    y_true,
    y_pred,
    eps=1e-6,
    alpha=0.5,
    class_weights=None,
    background_idx=0,
    invert_to_class=1,     # kept for signature compatibility; no longer used
):
    """
    Vectorised CE + Dice for multiclass segmentation.

    Fixes vs original:
      - No fake-target inversion branch: it had a loss floor of log(C-1) that
        never reached zero and pushed the background logit toward +inf forever.
        Standard CE on background patches is already -log(p_bg), which is the
        intended objective and correctly reaches zero.
      - Dice averaged over classes PRESENT in each patch only, so absent
        classes no longer dominate the Dice term. This also makes background
        patches reduce naturally to dice(p_bg, 1).
      - No Python loop over the batch; AMP-safe.

    y_true: [B, H, W] labels in [0..C-1]
    y_pred: [B, C, H, W] raw logits
    """
    B, C, H, W = y_pred.shape
    y_true    = y_true.long()
    log_probs = F.log_softmax(y_pred.float(), dim=1)
    probs     = log_probs.exp()

    # ---- CE (per-sample weighted mean, matching reduction='mean' semantics) ----
    ce_map = F.nll_loss(log_probs, y_true, weight=class_weights,
                        reduction='none')                       # [B, H, W]
    if class_weights is not None:
        w_map = class_weights.to(y_pred.device)[y_true]         # [B, H, W]
        ce = ce_map.flatten(1).sum(1) / w_map.flatten(1).sum(1).clamp(min=eps)
    else:
        ce = ce_map.flatten(1).mean(1)                          # [B]

    # ---- Dice over present classes only ----
    one_hot = F.one_hot(y_true, C).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    inter   = (probs * one_hot).flatten(2).sum(2)               # [B, C]
    union   = probs.flatten(2).sum(2) + one_hot.flatten(2).sum(2)
    dice_pc = 1.0 - (2.0 * inter + eps) / (union + eps)         # [B, C]

    present = one_hot.flatten(2).sum(2) > 0                     # [B, C]
    dice    = (dice_pc * present).sum(1) / present.sum(1).clamp(min=1)  # [B]

    return (alpha * ce + (1.0 - alpha) * dice).mean()

# def loss_with_inversion(y_true, y_pred, eps=1e-6, alpha=0.5, fp_weight=10.0):
#     """
#     Combined BCE + Dice loss with background-patch inversion.
    
#     Args:
#         y_true: [B, H, W] tensor with binary ground truth (0 or 1)
#         y_pred: [B, H, W] tensor with raw logits (not yet sigmoid)
#         eps: small value to avoid division by zero
#         alpha: weight for BCE (1 - alpha for Dice)
#     Returns:
#         scalar loss
#     """
#     losses = []

#     for i in range(y_true.shape[0]):
#         yi = y_true[i].float()
#         yhi = y_pred[i]  # still logits
#         # print(yi.min().item(), yi.max().item(), yi.unique())

#         # check if this is a background-only patch
#         if yi.sum() == 0:
#             # Invert: 0 → 1, 1 → 0
#             yi_inv = 1 - yi
#             yhi_inv = -torch.sigmoid(yhi) + 1  # equivalent to 1 - sigmoid

#             # compute BCE loss
#             bce = F.binary_cross_entropy(yhi_inv, yi_inv, reduction='mean')

#             # compute Dice loss
#             intersection = (yi_inv * yhi_inv).sum()
#             union = yi_inv.sum() + yhi_inv.sum()
#             dice = 1 - (2 * intersection + eps) / (union + eps)
#         else:
#             # Normal case
#             yhs = torch.sigmoid(yhi)

#             bce = F.binary_cross_entropy(yhs, yi, reduction='mean')
#             # Weight BCEloss
#             weights = torch.where(yi == 0, fp_weight, 1.0)  # amplify background penalty
#             bce = F.binary_cross_entropy(yhs, yi, weight=weights, reduction='mean')
#             intersection = (yi * yhs).sum()
#             union = yi.sum() + yhs.sum()
#             dice = 1 - (2 * intersection + eps) / (union + eps)

#         loss = alpha * bce + (1 - alpha) * dice
#         losses.append(loss)

#     return torch.stack(losses).mean()

# import torch
# import torch.nn.functional as F

# def loss_with_inversion_multiclass(
#     y_true,
#     y_pred,
#     eps=1e-6,
#     alpha=0.5,
#     class_weights=None,
#     background_idx=0,
#     invert_to_class=1,
# ):
#     """
#     Combined CrossEntropy + Dice loss with background-patch inversion for multi-class segmentation.

#     Args:
#         y_true: [B, H, W] tensor with ground truth labels in [0..C-1]
#         y_pred: [B, C, H, W] tensor with raw logits
#         eps: small value to avoid division by zero
#         alpha: weight for CE (1 - alpha for Dice)
#         class_weights: optional tensor [C] with weights for each class (for CE)
#         background_idx: which class index is background
#         invert_to_class: class index used as "fake foreground" when patch is pure background
#                         (must be != background_idx). Default assumes class 1 exists.
#     Returns:
#         scalar loss
#     """
#     B, C, H, W = y_pred.shape
#     if C < 2:
#         raise ValueError("Inversion requires at least 2 classes (C >= 2).")
#     if not (0 <= background_idx < C):
#         raise ValueError(f"background_idx must be in [0, {C-1}]")
#     if not (0 <= invert_to_class < C):
#         raise ValueError(f"invert_to_class must be in [0, {C-1}]")
#     if invert_to_class == background_idx:
#         raise ValueError("invert_to_class must be different from background_idx.")

#     losses = []

#     for i in range(B):
#         yi = y_true[i].long()     # [H, W]
#         yhi = y_pred[i]           # [C, H, W], logits

#         # --- Check if patch is pure background ---
#         if torch.all(yi == background_idx):
#             # Fake target: pick a non-background class everywhere
#             yi_inv = torch.full_like(yi, fill_value=invert_to_class)

#             # Build "inverted" logits:
#             # - Make background logit very unfavorable (using -bg_logit)
#             # - Set all non-background logits to 0 (uniform among foreground)
#             bg_logit = yhi[background_idx]                    # [H, W]
#             inv_logits = torch.zeros_like(yhi)                # [C, H, W]
#             inv_logits[background_idx] = -bg_logit            # discourage background

#             ce = F.cross_entropy(
#                 inv_logits.unsqueeze(0),
#                 yi_inv.unsqueeze(0),
#                 weight=class_weights,
#             )

#             probs = torch.softmax(inv_logits, dim=0)          # [C, H, W]
#             probs_flat = probs.reshape(C, -1)
#             yi_onehot = F.one_hot(yi_inv, num_classes=C).permute(2, 0, 1).reshape(C, -1).float()

#         else:
#             # Normal patch: standard CE + multi-class Dice
#             ce = F.cross_entropy(
#                 yhi.unsqueeze(0),
#                 yi.unsqueeze(0),
#                 weight=class_weights,
#             )

#             probs = torch.softmax(yhi, dim=0)                 # [C, H, W]
#             probs_flat = probs.reshape(C, -1)
#             yi_onehot = F.one_hot(yi, num_classes=C).permute(2, 0, 1).reshape(C, -1).float()

#         intersection = (probs_flat * yi_onehot).sum(dim=1)
#         union = probs_flat.sum(dim=1) + yi_onehot.sum(dim=1)
#         dice_per_class = 1 - (2 * intersection + eps) / (union + eps)
#         dice = dice_per_class.mean()

#         loss = alpha * ce + (1 - alpha) * dice
#         losses.append(loss)

#     return torch.stack(losses).mean()


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

loss_ce = nn.CrossEntropyLoss()

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


scaler = torch.amp.GradScaler("cuda")

def prep_masks(masks, num_class):
    """Single place for mask preprocessing — used by BOTH train and val."""
    if num_class == 1:
        return ((masks != 0) & (masks != 8)).float()
    masks = masks.clone()
    masks[masks == 8] = 0
    return masks.clamp(0, num_class - 1).long()

best_val_iou, stale, PATIENCE = 0.0, 0, 15

for epoch in range(wandb.config.epochs):
    dataset.current_epoch = epoch
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"epoch {epoch}"):
        images = images.to(device, non_blocking=True)
        masks  = prep_masks(masks.to(device, non_blocking=True), num_class)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            preds = model(images)
            if num_class == 1:
                preds = preds.squeeze(1)
                loss  = loss_with_inversion(masks, preds)
            else:
                loss  = loss_ce(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ── Validation ──
    model.eval()
    val_iou = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = prep_masks(masks.to(device, non_blocking=True), num_class)  # ← fixes bugs 1 & 2

            preds = model(images)
            if num_class == 1:
                preds_bin = torch.sigmoid(preds.squeeze(1)) > 0.5
                val_iou  += iou_metric(preds_bin, masks.int()).item()
            else:
                val_iou  += mean_iou_from_logits(preds.float(), masks, num_class).item()

    avg_val_iou = val_iou / len(val_loader)

    wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_iou": avg_val_iou})
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_val_iou:
        best_val_iou, stale = avg_val_iou, 0
        torch.save(model.state_dict(), save_path)
        print(f"  Saved new best (IoU {best_val_iou:.4f})")
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best {best_val_iou:.4f})")
            break

# for epoch in range(wandb.config.epochs):
#     dataset.current_epoch = epoch 
#     model.train()
#     train_loss = 0
#     for i, (images, masks) in enumerate(tqdm(train_loader)):
#         # print(images.shape, masks.shape)
#         images = images.to(device, non_blocking=True) 
#         masks = masks.to(device, non_blocking=True).float() 
#         # bh_ids = bh_ids.to(device, non_blocking=True)
#         if num_class == 1:
#             masks = ((masks != 0) & (masks != 8)).float()  # binary mask for BCE 
#             # masks = masks.unsqueeze(1)  # only tectonic fault as positive class, ignore others
#             # masks = (masks == 1).float()  # only tectonic fault as positive class, ignore others
#         # Create the input for attention       

#         preds = model(images).squeeze(1)
#         if num_class > 1:
#             # If multi-class
#             masks[masks == 8] = 0
#             masks = masks.long()
#             # loss = loss_with_inversion_multiclass(masks, preds, class_weights=torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 10.0]).to(device))
#             loss = loss_ce(preds, masks)
#         else:
#             loss = loss_with_inversion(masks, preds)
#             # loss = loss_ce(preds, masks)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     avg_train_loss = train_loss / len(train_loader)

#     # Evaluation
#     model.eval()
#     val_iou = 0
#     with torch.no_grad():
#         for images, masks in val_loader:
#             images = images.to(device, non_blocking=True) 
#             masks = masks.to(device, non_blocking=True).float()
#             # bh_ids = bh_ids.to(device, non_blocking=True)
#             preds = model(images).squeeze(1)
#             preds_bin = torch.sigmoid(preds) > 0.5
#             # If multi-class
#             if num_class > 1:
#                 val_iou += mean_iou_from_logits(preds, masks, num_class).item() 
#             else:            
#                 val_iou += iou_metric(preds_bin, masks.int()).item()

#     avg_val_iou = val_iou / len(val_loader)

#     # wandb logging
#     wandb.log({
#         "epoch": epoch,
#         "train_loss": avg_train_loss,
#         "val_iou": avg_val_iou
#     })

#     print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

#     # Save best model
#     if avg_val_iou > best_val_iou:
#         best_val_iou = avg_val_iou
#         torch.save(model.state_dict(), save_path)
#         print(f"Saved new best model at epoch {epoch} with IoU {best_val_iou:.4f}")

