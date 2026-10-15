import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import os
import cv2
from unetwithpe import UNetWithPE, UNetWithPE_BH
from finetune_train import SimpleInternImageSeg

# from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp

num_class = 8
add_coords = False
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=num_class)
model_path = "logs/general-borehole-cnn/Unet-resnet34-directmulti-onlyrgb-dcn-ep200-lr0.0001-260529-trace0.7.pth"

# model = UNetWithPE(encoder_name="resnet34", encoder_weights="imagenet",in_channels=6, classes=num_class, use_dcn=True)
# model_path = "logs/general-borehole-cnn/Unet-resnet34-42bh-binary-dcn-ep200-lr0.0001-260414-trace0.7_multiscale_binary.pth"

# model = UNetWithPE(encoder_name="resnet34", encoder_weights="imagenet",in_channels=6, classes=num_class, use_dcn=True,)
# # direct multi with pe inject fullly, normalized depth to 0-1
# model_path = "logs/general-borehole-cnn/Unet-resnet34-directmulti-allpeinjection-dcn-ep200-lr0.0001-260529-trace0.7.pth"
# Direct multi model with pe
# model_path = "logs/general-borehole-cnn/Unet-resnet34-directmulti-dcn-ep200-lr0.0001-260528-trace0.7.pth"
# model_path = "logs/general-borehole-cnn/Unet-mit_b2-41bh-binary-dcn-ep200-lr0.0001-260415-trace0.7.pth"
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.001-260128-trace0.7_multiscale_inversionloss_dcn.pth"
# model_path = "logs/general-borehole-cnn/Unet-resnet34-binary-dcn-ep200-lr0.001-260425-trace0.7.pth"
# model_path = "logs/general-borehole-cnn/Unet-resnet34-binary-dcn-ep200-lr0.001-260504-trace0.7.pth" 
# model_path = "logs/general-borehole-cnn/Unet-mit_b2-binary-dcn-ep200-lr0.001-260422-trace0.7.pth"
# Load the weights
# Binary model with inversion loss, no dcn
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.001-260128-trace0.7_multiscale_inversionloss_nodcn.pth"
# Direct multi model
# model_path = "logs/general-borehole-cnn/Unet-resnet34-42bh-binary-dcn-ep200-lr0.0001-260408-trace0.7_multiscale_multiclass.pth"
# This is the important one
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-multi-dcn-ep150-lr0.001-260127-trace0.7_multiscale_celoss.pth"
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.0001-260330-trace0.7_multiscale_fault.pth"
# Use ssl pretrain
# model_path = "logs/ssl_finetune/best_finetune.pth"
# Only tectonic fault binary model with inversion loss, with dcn
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep50-lr0.0001-260302-trace0.7_multiscale_inversiveloss_hardnegmining_512.pth"
# Soft label binary model with soft binary loss, with dcn
# model_path = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.0001-260311-trace0.7_multiscale_softlabel.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
model.cuda()

################################################################ Fintune InternImageSeg
# num_class = 9
# add_coords = False
# device = "cuda" if torch.cuda.is_available() else "cpu"

# ckpt = torch.load("logs/internimage/internimage_borehole_best.pt", map_location=device)

# model = SimpleInternImageSeg(
#     model_name=ckpt["backbone_name"],
#     num_classes=ckpt["num_classes"],
# ).to(device)
# model.load_state_dict(ckpt["model_state_dict"])
# model.eval()


print("Model loaded successfully.")

# ==========================================================
# CONFIGURATION
# ==========================================================

# Example parameters
# model = YourSegmentationModel().cuda().eval()
# model.load_state_dict(torch.load("your_model_weights.pth"))
# image_path = "../data/Labeled Data Myriam/PF/BPF3_OPTV.tif"
# output_path = "../data/general-boreholes-output/BPF3_overlap_avg_pred.png"
# patch_size = 512
# stride = 256  # overlap = patch_size - stride

# ==========================================================
# UTILITIES
# ==========================================================

def normalize_patch(patch):
    """Normalize an RGB patch using ImageNet mean/std."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (patch - mean) / std


def save_mask(mask, output_path):
    """Save binary mask as grayscale PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
    print(f"✅ Mask saved to {output_path}")


def get_wrapped_patch(image, x, y, patch_size, h, w):
    """Extract patch with horizontal wrap-around if x+patch_size > w."""
    if x + patch_size <= w:
        return image[y:y+patch_size, x:x+patch_size]
    else:
        right = image[y:y+patch_size, x:w]
        left = image[y:y+patch_size, 0:(x+patch_size - w)]
        return np.hstack((right, left))


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================

@torch.no_grad()
def predict_large_image(model, image_path, patch_size=512, stride=256, wrap_horizontal=True):
    """
    Predict a large unwrapped borehole wall image with overlapping sliding windows.

    - Horizontal direction is treated as periodic (wrap-around).
    - Vertical direction is normal (no wrapping).
    - Overlapping regions are averaged to smooth predictions.
    """
    model.eval().cuda()

    # Load image and normalize to [0, 1]
    image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h, w, c = image.shape
    print(f"📷 Image loaded: {w}x{h}")

    # Pad vertically (if height is not divisible by patch_size)
    pad_h = (patch_size - h % patch_size) % patch_size
    padded_image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)), mode="reflect")
    padded_h = padded_image.shape[0]

    # Accumulators for averaging
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    pred_map = np.zeros((h, w), dtype=np.uint8)

    # Sliding window inference
    for y in tqdm(range(0, padded_h - patch_size + 1, stride), desc="Predicting"):
        for x in range(0, w, stride):
            # Extract patch (with horizontal wrap)
            if wrap_horizontal:
                patch = get_wrapped_patch(padded_image, x, y, patch_size, padded_h, w)
            else:
                patch = padded_image[y:y+patch_size, x:min(x+patch_size, w)]

            patch = normalize_patch(patch)

            patch = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().cuda()  # (1,3,H,W)

            # --- add coordinates like in training ---
            if add_coords:
                ph, pw = patch.shape[2:]
                yy, xx = torch.meshgrid(
                    torch.arange(ph, dtype=torch.float32, device=patch.device),
                    torch.arange(pw, dtype=torch.float32, device=patch.device),
                    indexing="ij"
                )
                yy = (yy + y) / h  # normalize by full height
                # xx = (xx + x) / w  # normalize by full width (circular coordinate)
                # MAX_DEPTH_PX = 100000
                # if "BFS_B15_bottom" in image_path:
                #     yy = (yy + y+36283) / MAX_DEPTH_PX
                # else:
                #     yy = (yy + y) / MAX_DEPTH_PX
                xx = (xx + x) / w  
                depth_enc = yy.unsqueeze(0)  # (1,H,W)
                angle_enc = torch.stack([torch.sin(2 * torch.pi * xx),
                                         torch.cos(2 * torch.pi * xx)], dim=0)  # (2,H,W)
                pe = torch.cat([depth_enc, angle_enc], dim=0).unsqueeze(0)  # (1,3,H,W)
                patch = torch.cat([patch, pe], dim=1)  # (1,6,H,W)

            # --- inference ---
            bh_id_tensor = torch.tensor([42], dtype=torch.long)
            logits = model(patch)
            # print(logits.shape)
            # inp = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

            # # Model inference
            # logits = model(inp)
            if num_class > 1:
                probs = torch.softmax(logits, dim=1)[0,0]  # (H,W) prob of class 1
                probs = probs.cpu().numpy()
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                # Determine region in original image
                y1 = min(y + patch_size, h)
                patch_h = y1 - y
                # Horizontal wrap-around accumulation
                if x + patch_size <= w:
                    x1 = x + patch_size
                    pred_map[y:y1, x:x1] = pred[0,:patch_h, :patch_size]
                    prob_sum[y:y1, x:x1] += probs[:patch_h, :patch_size]
                    count[y:y1, x:x1] += 1
                else:
                    right_w = w - x
                    left_w = patch_size - right_w
                    pred_map[y:y1, x:w] = pred[0,:patch_h, :right_w]
                    prob_sum[y:y1, x:w] += probs[:patch_h, :right_w]
                    count[y:y1, x:w] += 1
                    pred_map[y:y1, 0:left_w] = pred[0,:patch_h, right_w:]
                    prob_sum[y:y1, 0:left_w] += probs[:patch_h, right_w:]
                    count[y:y1, 0:left_w] += 1


            else:
                if logits.ndim == 3:
                    logits = logits.unsqueeze(1)  # (B,1,H,W)
                probs = torch.sigmoid(logits)
                probs = F.interpolate(probs, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
                probs = probs.squeeze().cpu().numpy()  # (H,W)

                # Determine region in original image
                y1 = min(y + patch_size, h)
                patch_h = y1 - y

                # Horizontal wrap-around accumulation
                if x + patch_size <= w:
                    x1 = x + patch_size
                    prob_sum[y:y1, x:x1] += probs[:patch_h, :patch_size]
                    count[y:y1, x:x1] += 1
                else:
                    right_w = w - x
                    left_w = patch_size - right_w
                    prob_sum[y:y1, x:w] += probs[:patch_h, :right_w]
                    count[y:y1, x:w] += 1
                    prob_sum[y:y1, 0:left_w] += probs[:patch_h, right_w:]
                    count[y:y1, 0:left_w] += 1

    # Average overlapping probabilities
    full_pred = prob_sum / np.maximum(count, 1e-8)

    # Binary mask
    binary_mask = (full_pred > 0.5).astype(np.uint8)

    return binary_mask, full_pred, pred_map


# ==========================================================
# EXAMPLE MAIN
# ==========================================================

if __name__ == "__main__":
    # Replace this with your actual model
    # model = YourSegmentationModel().cuda().eval()
    # model.load_state_dict(torch.load("your_model.pth"))

    borehole_name = "BCL11"
    experiment_name= "CL"
    image_path = f"data/Labeled Data Myriam/{experiment_name}/{borehole_name}_OPTV.tif"
    # make the patch size same as image width
    img_width = Image.open(image_path).size[0]
    patch_size = img_width
    stride = patch_size  # 50% overlap

    binary_mask, full_pred, pred_map = predict_large_image(
        model,
        image_path,
        patch_size=patch_size,
        stride=stride,
        wrap_horizontal=True  # important for cylindrical images
    )

    # Save results
    output_dir = "data/general-boreholes-output/directmulti"
    os.makedirs(output_dir, exist_ok=True)
    if num_class == 1:
        cv2.imwrite(
            f"{output_dir}/{borehole_name}_{patch_size}_onlyPEskipfusion_pred_binary_dcn_inversiveloss.png",
            (full_pred * 100).astype(np.uint8)
        )
        # save_mask(binary_mask, f"{output_dir}/{borehole_name}_{patch_size}_onlyPEskipfusion_pred_mask_dcn_inversionloss.png")
    else:
        cv2.imwrite(
            f"{output_dir}/{borehole_name}.png",
            (pred_map).astype(np.uint8)
        )
    # save_mask(binary_mask, f"{output_dir}/{borehole_name}_{patch_size}_onlyPEskipfusion_pred_mask.png")

    print("✅ Inference completed.")
