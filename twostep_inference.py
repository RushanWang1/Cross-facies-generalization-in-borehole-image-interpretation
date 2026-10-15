import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import os
import cv2
from unetwithpe import UNetWithPE

from utils import tent_adapt, build_pe

# from segmentation_model_cnn_twosteps import FrozenBinaryGuidedMulti

# Build a few patches from the new borehole for adaptation
def sample_patches_for_tent(image, n=20, patch_size=256, pe_channels=3):
    h, w    = image.shape[:2]
    patches = []
    for _ in range(n):
        y1  = np.random.randint(0, max(1, h - patch_size))
        x1  = np.random.randint(0, max(1, w - patch_size))
        patch = image[y1:y1+patch_size, x1:x1+patch_size]
        img_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        pe    = build_pe(x1, y1, h, w, patch_size, patch_size)
        x     = torch.cat([img_t, pe], dim=0).unsqueeze(0)
        patches.append(x)
    return patches

class FrozenBinaryGuidedMulti_new(nn.Module):
    def __init__(self, binary_model: nn.Module, num_classes: int, in_channels: int,
                 encoder_name="mit_b2", encoder_weights="imagenet",
                 pe_channels=3, binary_weight=1.0):
        super().__init__()
        self.binary = binary_model
        self.binary.train()                  # allow training
        for p in self.binary.parameters():
            p.requires_grad = True           # gradients flow through stage 1

        self.pe_channels = pe_channels
        self.rgb_channels = 3
        self.binary_weight = binary_weight

        self.input_proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        nn.init.eye_(self.input_proj.weight.squeeze(-1).squeeze(-1))

        self.refiner = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )

        self.pe_projs = nn.ModuleList([
            nn.Conv2d(pe_channels, c, kernel_size=1) if c > 0 else None
            for c in self.refiner.encoder.out_channels
        ])
        self.bin_projs = nn.ModuleList([
            nn.Conv2d(1, c, kernel_size=1) if c > 0 else None
            for c in self.refiner.encoder.out_channels
        ])

        self.gate_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        rgb     = x[:, :3]
        pe_full = x[:, 3:]

        # No torch.no_grad() — gradients flow back into stage 1
        bin_logits = self.binary(x)

        bin_prob   = torch.sigmoid(bin_logits)
        confidence = (bin_prob - 0.5).abs() * 2
        bin_guided = bin_prob * confidence * self.binary_weight

        x_proj   = self.input_proj(rgb)
        features = self.refiner.encoder(x_proj)

        pe_feats = []
        for i, f in enumerate(features):
            if self.pe_projs[i] is None:
                pe_feats.append(f)
                continue
            _, _, h, w = f.shape

            pe_scaled  = F.interpolate(pe_full, size=(h, w), mode="bilinear", align_corners=False)
            pe_proj    = self.pe_projs[i](pe_scaled)

            bin_scaled = F.interpolate(bin_guided, size=(h, w), mode="bilinear", align_corners=False)
            bin_proj   = self.bin_projs[i](bin_scaled)

            pe_feats.append(f + pe_proj + bin_proj)

        decoder_output = self.refiner.decoder(pe_feats)
        mc_logits      = self.refiner.segmentation_head(decoder_output)

        gate      = torch.sigmoid(bin_prob * self.gate_temp.clamp(0.1, 2.0))
        mc_logits = mc_logits * gate

        return mc_logits

class FrozenBinaryGuidedMulti(nn.Module):
    def __init__(self, binary_model: nn.Module, num_classes: int, in_channels: int,
                 encoder_name="resnet34", encoder_weights=None):
        super().__init__()
        self.binary = binary_model
        self.binary.eval()
        for p in self.binary.parameters():
            p.requires_grad = False

        self.refiner = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels + 1,  # 7
            classes=num_classes
        )

    def forward(self, x):
        with torch.no_grad():
            bin_logits = self.binary(x)              # [B,1,H,W]
            if bin_logits.ndim == 3:
                bin_logits = bin_logits.unsqueeze(1) # safety
        # x_rgb = x[:,:3,:,:]               # [B,3,H,W]
        x_guided = torch.cat([x, bin_logits], dim=1) # [B,7,H,W]
        return self.refiner(x_guided)                # [B,9,H,W]

class FrozenBinaryGuidedMulti_gated(nn.Module):
    def __init__(self, binary_model: nn.Module, num_classes: int, in_channels: int,
                 encoder_name="resnet34", encoder_weights="imagenet", gate_alpha=0.5):
        super().__init__()
        self.binary = binary_model
        self.binary.eval()
        for p in self.binary.parameters():
            p.requires_grad = False

        self.gate_alpha = gate_alpha

        self.refiner = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels + 1,
            classes=num_classes
        )

    def forward(self, x):
        with torch.no_grad():
            bin_logits = self.binary(x)          # [B,1,H,W]

        bin_prob = torch.sigmoid(bin_logits)     # [B,1,H,W]

        x_gated = x * (1.0 + self.gate_alpha * bin_prob)
        x_guided = torch.cat([x_gated, bin_prob], dim=1)

        mc_logits = self.refiner(x_guided)
        return mc_logits #, bin_logits
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_class = 9
add_coords = False
binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-42bh-binary-dcn-ep200-lr0.0001-260407-trace0.7_multiscale.pth"
multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260407-trace0.5_trainboth_onlyrgb.pth"
###### Paper_final: 
# binary_ckpt = "logs/paper/Unetwithpe-resnet34-42bh-binary-dcn-ep200-lr0.001-260128-trace0.7_multiscale_inversionloss_dcn.pth"
# multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260129-trace0.7_trainboth.pth"
# multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260401-trace0.5_trainboth_focal_dice.pth"
# multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260319-trace0.5_trainboth_focal_dice.pth"
# multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260318-trace0.5_trainboth_ce_dice.pth"
# binary_ckpt = "logs/general-borehole-cnn/Unetwithpe-resnet34-42bh-binary-dcn-ep50-lr0.0001-260302-trace0.7_multiscale_inversiveloss_hardnegmining_512.pth"
# multi_ckpt = "logs/paper/twostep-ep150-lr0.001-260302-trace0.7_trainboth_hardnegmining.pth"
# # New 46 borehole
# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-42bh-binary-dcn-ep200-lr0.0001-260414-trace0.7_multiscale_binary.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260414-trace0.5_trainboth_onlyrgb.pth"


# mit_b2 encoder
# binary_ckpt = "logs/general-borehole-cnn/Unet-mit_b2-binary-dcn-ep200-lr0.001-260422-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260506-trace0.7_trainboth_mit_b2.pth"
# Efficientnet-b4 encoder
# binary_ckpt = "logs/general-borehole-cnn/Unet-efficientnet-b4-41bh-binary-dcn-ep200-lr0.0001-260426-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260506-trace0.7_trainboth_efficientnet-b4.pth"
# resnet34 encoder
# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-41bh-binary-dcn-ep200-lr0.0001-260427-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260507-trace0.7_trainboth_resnet34.pth"
# Weighted celoss + norm depth (0,1)
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260601-trace0.7_trainboth_resnet34_normdepth.pth"
# Weighted celoss + maxnorm depth
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260601-trace0.7_trainboth_resnet34_maxnormdepth.pth"

# 46 boreholes resnet34 encoder
# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-46bh-binary-dcn-ep200-lr0.0001-260520-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260521-trace0.7_trainboth_resnet34_46bh.pth"
# Trained only on lower shaly label (resnet34 encoder)
# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-lowershaly-binary-dcn-ep200-lr0.0001-260522-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260527-trace0.7_trainboth_resnet34.pth"

# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-41bh-binary-dcn-ep200-lr0.0001-260416-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260420-trace0.7_trainboth_resnet34.pth"

# facies balance
# binary_ckpt = "logs/general-borehole-cnn/Unet-resnet34-binary-faciesbalance-dcn-ep200-lr0.0001-260722-trace0.7.pth"
# multi_ckpt = "logs/paper/twostep-ep250-lr0.001-260722-trace0.7_trainboth_resnet34_faciesbalance.pth"

encoder_name = "mit_b2" if "mit_b2" in binary_ckpt else ("efficientnet-b4" if "efficientnet-b4" in binary_ckpt else "resnet34")

# binary = UNetWithPE(
#     encoder_name=encoder_name,
#     encoder_weights=None,
#     in_channels=6,
#     classes=1,
#     use_dcn=True,
#     return_features = False
# ).to(device)

binary = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type=None
).to(device )

# binary.load_state_dict(torch.load(multi_ckpt, map_location=device))
# binary.eval()
# for p in binary.parameters():
#     p.requires_grad = False


model = FrozenBinaryGuidedMulti(
    binary_model=binary,
    num_classes=num_class,
    in_channels=3,
    encoder_name=encoder_name,
    encoder_weights=None,
).to(device)

# load wrapper weights (this will also try to load binary.* but it’s ok because architecture matches)
model.load_state_dict(torch.load(multi_ckpt, map_location=device), strict=True)
model.eval()
print("Two-step model loaded successfully.")


def normalize_patch(patch):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (patch - mean) / std

def get_wrapped_patch(image, x, y, patch_size, h, w):
    if x + patch_size <= w:
        return image[y:y+patch_size, x:x+patch_size]
    else:
        right = image[y:y+patch_size, x:w]
        left = image[y:y+patch_size, 0:(x+patch_size - w)]
        return np.hstack((right, left))

@torch.no_grad()
def predict_large_image_two_step(model, image_path, patch_size=512, stride=256, wrap_horizontal=True, return_class=True, class_num = 1, add_coords=add_coords):
    """
    Two-step inference with overlapping sliding windows + horizontal wrap-around.

    model(x6) -> (bin_logits, mc_logits)
    """
    model.eval()

    image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = image.shape
    print(f"Image loaded: {w}x{h}")

    pad_h = (patch_size - h % patch_size) % patch_size
    padded_image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)), mode="reflect")
    padded_h = padded_image.shape[0]

    # outputs
    pred_map = np.zeros((h, w), dtype=np.uint8)

    # stage-1 binary prob accumulation (optional but useful)
    bin_prob_sum = np.zeros((h, w), dtype=np.float32)
    bin_count = np.zeros((h, w), dtype=np.float32)

    for y in tqdm(range(0, padded_h - patch_size + 1, stride), desc="Predicting"):
        for x in range(0, w, stride):
            # patch extraction
            if wrap_horizontal:
                patch = get_wrapped_patch(padded_image, x, y, patch_size, padded_h, w)
            else:
                patch = padded_image[y:y+patch_size, x:min(x+patch_size, w)]

            patch = normalize_patch(patch)
            patch = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)  # [1,3,ps,ps]

            # add coords (same as training)
            if add_coords:
                ph, pw = patch.shape[2:]
                yy, xx = torch.meshgrid(
                    torch.arange(ph, dtype=torch.float32, device=patch.device),
                    torch.arange(pw, dtype=torch.float32, device=patch.device),
                    indexing="ij"
                )
                MAX_DEPTH_PX = 100000
                # if "BFS_B15_bottom" in image_path:
                #     yy = (yy + y+36283) / MAX_DEPTH_PX
                # else:
                #     yy = (yy + y) / MAX_DEPTH_PX
                yy = (yy + y) / h  # normalize by full height
                xx = (xx + x) / w  

                depth_enc = yy.unsqueeze(0)  # [1,H,W]
                angle_enc = torch.stack(
                    [torch.sin(2 * torch.pi * xx), torch.cos(2 * torch.pi * xx)],
                    dim=0
                )  # [2,H,W]
                pe = torch.cat([depth_enc, angle_enc], dim=0).unsqueeze(0)  # [1,3,H,W]
                x6 = torch.cat([patch, pe], dim=1)  # [1,6,H,W]
            else:
                # raise ValueError("This two-step pipeline expects PE (add_coords=True).")
                # print("No PE injected.")
                x6 = patch  # [1,3,H,W]

            # --- two-step forward ---
            mc_logits = model(x6)  # [1,C,H,W]
            if return_class:
                pred_patch = mc_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            else:
                mc_probs = torch.softmax(mc_logits, dim=1)
                prob_maps = mc_probs[0].detach().cpu().numpy()# [C, H, W], float32
                pred_patch = prob_maps[class_num] * 100  # e.g. return probability of class 1 as heatmap

            # (optional) get binary prob map directly from frozen binary model
            # bin_logits = model.binary(x6)  # [1,1,H,W]
            # # bin_prob = torch.sigmoid(bin_logits)[0, 0].detach().cpu().numpy()
            # pred_patch = bin_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # [ps,ps]

            # paste into full map with wrap handling
            y1 = min(y + patch_size, h)
            patch_h = y1 - y

            if x + patch_size <= w:
                x1 = x + patch_size
                pred_map[y:y1, x:x1] = pred_patch[:patch_h, :patch_size]

                # bin_prob_sum[y:y1, x:x1] += bin_prob[:patch_h, :patch_size]
                bin_count[y:y1, x:x1] += 1
            else:
                right_w = w - x
                left_w = patch_size - right_w

                pred_map[y:y1, x:w] = pred_patch[:patch_h, :right_w]
                pred_map[y:y1, 0:left_w] = pred_patch[:patch_h, right_w:]

                # bin_prob_sum[y:y1, x:w] += bin_prob[:patch_h, :right_w]
                bin_count[y:y1, x:w] += 1
                # bin_prob_sum[y:y1, 0:left_w] += bin_prob[:patch_h, right_w:]
                bin_count[y:y1, 0:left_w] += 1

    # bin_prob_map = bin_prob_sum / np.maximum(bin_count, 1e-8)
    return pred_map #,bin_prob_map

if __name__ == "__main__":
    borehole_name = "BCL11"
    experiment_name = "CL"
    image_path = f"data/Labeled Data Myriam/{experiment_name}/{borehole_name}_OPTV.tif"
    # image_path = "GFZ_data/TSM01_bottom/TSM01_bottom_OTV_20Mai.tif"
    img_width = Image.open(image_path).size[0]
    # patch_size = img_width
    # stride = patch_size  # 50% overlap
    patch_size = 720
    stride = 360
    output_dir = "data/general-boreholes-output/onlyrgb_twostep"

    # ---- Load image for TENT patch sampling ----
    image_np = np.array(Image.open(image_path).convert("RGB"))  # uint8 [H, W, 3]
    h, w     = image_np.shape[:2]
    run_tent = False
    if run_tent:
        print("Sampling patches for TENT adaptation...")
        patches = sample_patches_for_tent(
            image_np,
            n=30,               # number of patches to adapt on
            patch_size=512,     # use 512 for TENT regardless of inference patch size
        )

        # TENT modifies a copy — original model weights unchanged
        model = tent_adapt(
            model,
            patches,
            device,
            n_steps=20,         # 10-30 steps is usually enough
            lr=1e-4,
        )
        print("TENT adaptation done.")


    pred_map = predict_large_image_two_step(
        model,
        image_path,
        patch_size=patch_size,
        stride=stride,
        wrap_horizontal=True,
        return_class=True,
        class_num=1,  # e.g. return probability of class 1 as heatmap; ignored if return_class=True
    )

    
    os.makedirs(output_dir, exist_ok=True)
    if run_tent:
        cv2.imwrite(
            f"{output_dir}/{borehole_name}_{patch_size}_two_step_pred_focaldiceloss_multiclass_trainboth_stride_{stride}_tent.png",
            pred_map.astype(np.uint8)
        )
    else:
        cv2.imwrite(
            # f"{output_dir}/{borehole_name}_two_step_pred_multiclass_trainboth_stride_{encoder_name}.png",
            f"{output_dir}/{borehole_name}_patch_{patch_size}.png",
            # "GFZ_data/TSM01_bottom/TSM01_bottom_two_step_pred.png",
            pred_map.astype(np.uint8)
        )

    # # Save mix class 1 and class 2
    # prob_class1 = (pred_map * 2.55).astype(np.uint8)  # [H, W]
    # prob_class2 = (pred_map2 * 2.55).astype(np.uint8)  # [H, W]
    # zeros = np.zeros_like(prob_class1)

    # # BGR format for OpenCV: class1=Red, class2=Green, overlap=Yellow
    # vis = np.stack([zeros, prob_class2, prob_class1], axis=-1)  # [H, W, 3]
    # cv2.imwrite(f"{output_dir}/{borehole_name}_{patch_size}_two_step_pred_celoss_multiclass_trainboth_stride_{stride}_mix.png", vis)

    # optional: save binary probability heatmap (0..255)
    # cv2.imwrite(
    #     f"{output_dir}/{borehole_name}_{patch_size}_two_step_celoss_binprob_trainonlyfeature.png",
    #     (np.clip(bin_prob_map, 0, 1) * 255).astype(np.uint8)
    # )

    print("Two-step inference completed.")
